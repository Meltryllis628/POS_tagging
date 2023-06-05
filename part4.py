import numpy as np

class StructuredPerceptron:
    def __init__(self,train_file, test_file):
        self.weight = {}
        
        self.pad_token = "#PAD#"
        
        self.train_data = self.__read_data(train_file)
        self.test_data = self.__read_data(test_file)
        
        self.labels = self.__get_labels(self.train_data)
        self.n_labels = len(self.labels)
        self.n_samples = len(self.train_data)
        self.label_dict = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        self.trainX,self.trainY = self.get_X_Y(self.train_data)
        self.testX,self.testY = self.get_X_Y(self.test_data)

    def __read_data(self,file_path):
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
    
    def __read_data(self,file_path):
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
                    try:
                        label = parts[1]
                    except:
                        label = "O"
                    words.append(word)
                    labels.append(label)
            if words:
                data[len(data)] = {'words': words, 'labels': labels}
        return data

    def __get_labels(self, data):
        labels = set()
        for sentence in data.values():
            labels.update(sentence['labels'])
        return sorted(labels)
    
    def get_X_Y(self,data):
        data = list(data.values())
        X = []
        Y = []

        for sentence in data:
            X.append(sentence['words'])
            Y_temp = []
            for label in sentence['labels']:
                Y_temp.append(self.label_dict[label])
            Y.append(Y_temp)
        return X,Y
    
    def __get_transition_matrix(self):
        # This method returns a transition matrix that represents the probabilities of transitioning 
        # from one label to another label.
        transition_matrix = np.array(
            [[self.get_weight(f"{i}:{j}") for j in range(self.n_labels)] for i in range(self.n_labels)])
        return transition_matrix
    
    def __get_emission_matrix(self,x):
        # This method returns an emission matrix that represents the probabilities of observing each 
        # label for each feature in the input sentence x.
        emission_matrix = np.array(
            [[sum(self.get_weight(f"{feature}{tag}") for feature in features)
              for tag in range(self.n_labels)] for features in self.get_features(x)])
        
        return emission_matrix
    
    def set_weight(self, key, delta):
        # This method sets the weight for the given key to be the sum of its current value and delta.
        self.weight[key] = self.weight.get(key, 0) + delta

    def get_weight(self, key, default=0):
        # This method gets the weight for the given key, returning default if the key is not in self.weight.
        if key not in self.weight:
            return default
        else:
            return self.weight[key]

    def get_features(self, x):
        # This method extracts a list of features for each token in the input sentence x.
        features = []
        for t in range(len(x)):
            prev2 = x[t-2] if t-2 >= 0 else self.pad_token
            prev1 = x[t-1] if t-1 >= 0 else self.pad_token
            cur = x[t]
            next1 = x[t+1] if t+1 < len(x) else self.pad_token
            next2 = x[t+2] if t+2 < len(x) else self.pad_token
            feature_list = ["1" + prev2, "2" + prev1, "3" + cur,"4" + next1, "5" + next2,"6" + cur + prev1, "7" + cur + next1,"8"+prev1+prev2,"9"+next1+next2]
            features.append(feature_list)
        return features

    def veterbi(self, sentence):
        # This method uses the Viterbi algorithm to find the most likely sequence of labels for the input sentence.
        transition_matrix = self.__get_transition_matrix()
        emission_matrix = self.__get_emission_matrix(sentence)
        pi= emission_matrix[0]
        path = []       
        for t in range(1, len(sentence)):
            # Calculate the probability of being in each label at time t, given the previous label.
            max_prob = pi.reshape(-1, 1) + transition_matrix + emission_matrix[t]
            # Find the label that maximizes the probability.
            prev_state = np.argmax(max_prob, axis=0)
            # Update the probability distribution for the current time step.
            pi = np.max(max_prob, axis=0)
            path.append(prev_state)
        # Trace back the most likely sequence of labels.
        res = [np.argmax(pi)]
        for p in reversed(path):
            res.append(p[res[-1]])
        return list(reversed(res))

    def train(self, epochs=10):
        for epoch in range(epochs):
            # For each training example (x, y) in the training data, make a prediction and update weights if necessary
            for _, (x, y) in enumerate(zip(self.trainX, self.trainY)):
                # Make a prediction for the current input sequence x using the Viterbi algorithm
                y_pred = self.veterbi(x)
                # If the predicted output y_pred is not the same as the true output y, update the weight vector
                if y_pred != y:
                    # Update the weights for the correct output y
                    self.update_weight(x, y, 1)
                    # Update the weights for the predicted output y_pred
                    self.update_weight(x, y_pred, -1)
            train_acc = self.accuracy(self.trainX, self.trainY)
            test_acc = self.accuracy(self.testX, self.testY)
            print(f"Epoch {epoch+1} completed, train accuracy is {train_acc}, test accuracy is {test_acc}")

    def update_weight(self, x, y, delta):
        # Iterate through each position in the input sequence x and its corresponding output label y
        for i, features in enumerate(self.get_features(x)):
            for feature in features:
                # For each feature of the input at position i, update the weight vector for the output label y[i] by delta
                self.set_weight(f"{feature}{y[i]}", delta)
        # Iterate through each adjacent pair of output labels in y and update the weight vector for the transition between them
        for i in range(1, len(x)):
            self.set_weight(f"{y[i-1]}:{y[i]}", delta)

    def predict(self, dataX):
        pred = []
        for x in dataX:
            y = self.veterbi(x)
            pred.append(y)
        return pred

    def accuracy(self, dataX, dataY):
        total_count = 0
        correct_count = 0
        pred_labels = self.predict(dataX)
        target_labels = dataY
            
        for pred_label,target_label in zip(pred_labels, target_labels):
            total_count+=len(pred_label)
            for j in range(len(pred_label)):
                if pred_label[j] == target_label[j]:
                    correct_count += 1
        accuracy = correct_count / total_count
        return accuracy
    
    def write_result(self, filename):
        pred_labels_raw =self.predict(self.testX)
        result = {}
        for i,sentence in enumerate(pred_labels_raw):
            labels = []
            for word in sentence:
                labels.append(self.id2label[word])
            result[i]=labels
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, labels in result.items():
                sentence = self.testX[idx]
                for word, label in zip(sentence, labels):
                    f.write("{} {}\n".format(word, label))
                f.write("\n")

    def write_result_2(self, another, filename):
        test_data = self.__read_data(another)
        testX,_ = self.get_X_Y(test_data)
        pred_labels_raw =self.predict(testX)
        result = {}
        for i,sentence in enumerate(pred_labels_raw):
            labels = []
            for word in sentence:
                labels.append(self.id2label[word])
            result[i]=labels
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, labels in result.items():
                sentence = testX[idx]
                for word, label in zip(sentence, labels):
                    f.write("{} {}\n".format(word, label))
                f.write("\n")


train_file = "EN\\train"
test_file = "EN\\dev.out"
test_file_2 = "Test\\EN\\test.in"
model = StructuredPerceptron(train_file, test_file)
model.train()
model.write_result("EN\\dev.p4.out")
model.write_result_2(test_file_2, "EN\\test.p4.out")

train_file_ = "FR\\train"
test_file_ = "FR\\dev.out"
test_file_2_ = "Test\\FR\\test.in"
model_ = StructuredPerceptron(train_file_, test_file_)
model_.train()
model_.write_result("FR\\dev.p4.out")
model_.write_result_2(test_file_2_, "FR\\test.p4.out")