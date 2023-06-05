import numpy as np
class HMM:
    def __init__(self, train_file, test_file):
        self.pad_token = "#PAD#"
        self.unk_token = "#UNK#"
        self.start_label = "<START>"
        self.stop_label = "<STOP>"

        self.train_data = self.__read_data(train_file)
        self.test_data = self.__read_data(test_file)

        self.labels = self.__get_labels(self.train_data)
        self.vocab = self.__get_vocab(self.train_data)
        self.vocab += [self.unk_token]
        self.n_labels = len(self.labels)
        self.n_vocab = len(self.vocab)
        self.n_samples = len(self.train_data)

        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.vocab_dict = {vocab: i for i, vocab in enumerate(self.vocab)}
        
        self.transition_matrix, self.labels = self.__get_transition_matrix(self.train_data)
        self.n_labels = len(self.labels)
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.feature_weights = np.zeros((self.n_vocab, self.n_labels))

        self.emission_matrix = self.__get_emission_matrix(self.train_data)
        self.max_len = 0
        
        # 保存训练数据和测试数据的输入和输出
        self.train_X = [sentence['words'] for sentence in self.train_data.values()]
        self.train_Y = [sentence['labels'] for sentence in self.train_data.values()]
        self.test_X = [sentence['words'] for sentence in self.test_data.values()]
        self.test_Y = [sentence['labels'] for sentence in self.test_data.values()]


        self.features = np.zeros((self.n_vocab, self.n_labels, self.n_labels))

    def size(self):
        return(self.n_samples, self.n_labels, self.n_vocab)
        
    def __read_data(self, file_path):
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        data[len(data)] = {'words': words, 'labels': labels}
                        words = []
                        labels = []
                else:
                    parts = line.split()
                    word = parts[0]
                    label = parts[1]
                    words.append(word)
                    labels.append(label)
            if words:
                data[len(data)] = {'words': words, 'labels': labels}
        return data


    def __get_vocab(self, data):
        vocab = set()
        for sentence in data.values():
            vocab.update(sentence['words'])
        return sorted(vocab)
    
    def __get_labels(self, data):
        labels = set()
        for sentence in data.values():
            labels.update(sentence['labels'])
        return sorted(labels)
    
    

    def __get_emission_matrix(self, data, smoothing=1.0):
        
        feature_counts = np.zeros((self.n_vocab, self.n_labels))
        emission_matrix = np.zeros((self.n_vocab, self.n_labels))
        label_counts = np.zeros(self.n_labels)
        
        for sentence in data.values():
            for i in range(len(sentence['words'])):
                word = sentence['words'][i]
                label = sentence['labels'][i]
                
                if word not in self.vocab:
                    word = self.unk_token
                
                feature_counts[self.vocab_dict[word]][self.label_dict[label]] += 1
                label_counts[self.label_dict[label]] += 1

        feature_counts[-1,:] = smoothing
        
        # smooth
        for i in range(self.n_labels):
            if i == self.label_dict[self.start_label] or i == self.label_dict[self.stop_label]:
                emission_matrix[:,i] = -np.inf
            else:
                emission_matrix[:,i] = np.log(feature_counts[:,i] / (label_counts[i] + smoothing))
        return emission_matrix
    
    def __get_transition_matrix(self, data):
        labels_with_boundary = [self.start_label] + self.labels + [self.stop_label]
        
        num_labels = len(labels_with_boundary)
        transition_counts = np.zeros((num_labels, num_labels))
        
        for sentence in data.values():
            labels = sentence['labels']
            labels = [self.start_label] + labels + [self.stop_label]
            for i in range(1, len(labels)):
                from_label = labels[i-1]
                to_label = labels[i]
                transition_counts[labels_with_boundary.index(from_label), labels_with_boundary.index(to_label)] += 1
        
        transition_weights = np.zeros((num_labels, num_labels))
        for i in range(num_labels):
            for j in range(num_labels):
                count = transition_counts[i, j]
                total_count = sum(transition_counts[i, :])
                transition_weights[i, j] = np.log(count / total_count)
        
        for i in (labels_with_boundary.index(self.start_label), labels_with_boundary.index(self.stop_label)):
            for j in (labels_with_boundary.index(self.start_label), labels_with_boundary.index(self.stop_label)):
                transition_weights[i][j] = -np.inf
        return transition_weights, labels_with_boundary
    
    def __estimate_with_emission_matrix(self, data):
        predicted = {}
        idx = 0
        for sentence in data:
            pred = []
            for word in sentence:
                if word not in self.vocab:
                    word = self.unk_token
                x = self.vocab_dict[word]
                y_list = self.emission_matrix[x, :]
                y = np.argmax(y_list)
                pred.append(y)
            predicted[idx] = pred
            idx += 1
        return predicted
    
    def __viterbi(self, data):
        predicted_y = {}
        idx = 0
        start = self.label_dict[self.start_label]
        stop = self.label_dict[self.stop_label]
        for sentence in data:
            n = len(sentence) + 2
            m = self.n_labels
            pi = np.log(np.zeros((n, m)))
            pi[0][start] = 0

            for i in range(len(sentence)):
                k = i+1
                x = sentence[i]
                if x not in self.vocab:
                    x = self.unk_token
                x = self.vocab_dict[x]
                for v in range(m):
                    for u in range(m):
                        log_p = pi[k-1][u]+self.transition_matrix[u][v]+self.emission_matrix[x][v]
                        if (log_p >= pi[k][v]):
                            pi[k][v] = log_p

            for u in range(m):
                log_p = pi[n-2][u]+self.transition_matrix[u][stop]
            if (log_p >= pi[n-1][stop]):
                pi[n-1][stop] = log_p

            pred_y = np.zeros((n), dtype = "int32")
            scores = np.log(np.zeros((n)))
            pred_y[n-1] = stop

            for v in range(m):
                log_p = pi[n-2][v]+self.transition_matrix[v][stop]
                if (log_p >= scores[n-2]):
                    scores[n-2] = log_p
                    pred_y[n-2] = v
            for i in range(n-3, 0, -1):
                for u in range(m):
                    log_p = pi[i][u]+self.transition_matrix[u][pred_y[i+1]]
                    if (log_p >= scores[i]):
                        scores[i] = log_p
                        pred_y[i] = u
            pred_y[0] = start
            predicted_y[idx] = list(pred_y[1:-1])
            idx += 1
        return predicted_y

    
    def eval_1(self, output_file):
        result = self.__estimate_with_emission_matrix(self.test_X)
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, labels in result.items():
                sentence = self.test_X[idx]
                for word, label in zip(sentence, labels):
                    f.write("{} {}\n".format(word, self.labels[label]))
                f.write("\n")

    def eval_2(self, output_file):
        result = self.__viterbi(self.test_X)
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, labels in result.items():
                sentence = self.test_X[idx]
                for word, label in zip(sentence, labels):
                    f.write("{} {}\n".format(word, self.labels[label]))
                f.write("\n")


train_file = "EN\\train"
test_file = "EN\\dev.out"
model = HMM(train_file, test_file)
output_file_1 = "EN\\dev.p1.out"
model.eval_1(output_file_1)
output_file_2 = "EN\\dev.p2.out"
model.eval_2(output_file_2)

train_file_ = "FR\\train"
test_file_ = "FR\\dev.out"
model = HMM(train_file_, test_file_)
output_file_1_ = "FR\\dev.p1.out"
model.eval_1(output_file_1_)
output_file_2_ = "FR\\dev.p2.out"
model.eval_2(output_file_2_)