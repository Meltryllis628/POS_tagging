import numpy as np
import re

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    words = []
    labels = []
    for line in data:
        if line.strip() == "":
            continue
        word, label = line.strip().split()
        words.append(word)
        labels.append(label)
    words = np.array(words)
    labels = np.array(labels)
    return words, labels
def read_data_sentence(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    words = []
    labels = []
    sentences = re.split('\n\n+', data)
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        words.append("")
        labels.append("START")
        lines = re.split('\n', sentence)
        for line in lines:
            word, label = line.split()
            words.append(word)
            labels.append(label)
        words.append("")
        labels.append("STOP")
    return words, labels
def estimate_transition_params(train_data, label2idx):

    m = len(label2idx)
    start = label2idx["START"]
    stop = label2idx["STOP"]

    transition_counts = np.zeros((m, m))
    transition_probs = np.zeros((m, m))


    for i in range(1, len(train_data)):
        prev_label = train_data[i-1]
        if (prev_label == stop):
            prev_label = start
        label = train_data[i]
        transition_counts[prev_label, label] += 1

    for i in range(m):
        transition_probs[:,i] = np.log(np.nan_to_num((transition_counts[:,i] / np.sum(transition_counts[:,i]))))

    # transition_probs = np.log(np.nan_to_num((transition_counts / np.sum(axis=1, keepdims=True))))

    return transition_probs
def label2idx(pred_labels, true_labels):
    labels = np.unique(np.concatenate((pred_labels, true_labels)))
    labels2int = dict((labels[i], i) for i in range(len(labels)))
    pred_labels_idx = np.array([labels2int[y] for y in pred_labels])
    true_labels_idx = np.array([labels2int[y] for y in true_labels])
    return labels2int, pred_labels_idx, true_labels_idx
def estimate_transition_params_2step(train_data, label2idx):
    m = len(label2idx)
    start = label2idx["START"]
    stop = label2idx["STOP"]

    transition_counts = np.zeros((m, m, m))# y-2 y-1 y 是正序的！
    transition_probs = np.zeros((m, m, m))

    for i in range(2, len(train_data)):
        label = train_data[i]
        prev_label = train_data[i-1]
        if (prev_label == start):
            prev_prev_label = start
        elif (prev_label == stop):
            label = stop
            prev_prev_label = train_data[i-2]
        else:
            prev_prev_label = train_data[i-2]
        transition_counts[prev_prev_label, prev_label, label] += 1

    # transition_counts[0, train_data[0]] += 1

    for i in range(m):
        transition_probs[:,:,i] = np.log(np.nan_to_num((transition_counts[:,:,i] / np.sum(transition_counts[:,:,i]))))

    # transition_probs = np.log(np.nan_to_num((transition_counts / transition_counts.sum(axis=2, keepdims=True))))

    return transition_probs
def viterbi_2(sentences, label2idx, emission_params, transition_params, print_req = False, lim = -23):
    m = len(label2idx)
    start = label2idx["START"]
    stop = label2idx["STOP"]
    predicted_y = {}
    for sentence_no in sentences:
        words = sentences[sentence_no]
        n = len(words) + 2
        pi = np.log(np.zeros((n, m, m))) # x y y-1 第二位和第三位是逆序的！ 第二位是当前的label， 第三位是前一个！
        pi[0][start][start] = 0
    # print("START", pi[0])

        k = 1
        x = words[0]
        if x not in emission_params:
            x = "#UNK#"
        for u in range(m-2):
            a = lim
            b = lim
            if u in emission_params[x]:
                a = np.log(emission_params[x][u])
            if transition_params[start][start][u] != -np.inf:
                b = transition_params[start][start][u]
            log_p = pi[k-1][start][start] + a + b
            if (log_p>pi[k][u][start]):
                pi[k][u][start] = log_p
        k = 2
        x = words[1]
        if x not in emission_params:
            x = "#UNK#"
        for v in range(m-2):
            for u in range(m-2):
                a = lim
                b = lim
                if v in emission_params[x]:
                    a = np.log(emission_params[x][v])
                if transition_params[start][u][v] != -np.inf:
                    b = transition_params[start][u][v]
                log_p = pi[k-1][u][start] + a + b
                if (log_p >= pi[k][v][u]):
                    pi[k][v][u] = log_p
        for i in range(2, len(words)):
            k = i+1
            x = words[i]
            if x not in emission_params:
                x = "#UNK#"
            for v in range(m-2):
                for u in range(m-2):
                    for w in range(m-2):
                        a = lim
                        b = lim
                        if v in emission_params[x]:
                            a = np.log(emission_params[x][v])
                        if transition_params[w][u][v] != -np.inf:
                            b = transition_params[w][u][v]
                        log_p = pi[k-1][u][w] + a + b
                        if (log_p >= pi[k][v][u]):
                            pi[k][v][u] = log_p
        k = len(words)+1
        for u in range(m-2):
            for w in range(m-2):
                b = -50
                if transition_params[w][u][stop] != -np.inf:
                    b = transition_params[w][u][stop]
                log_p = pi[k-1][u][w] + b
                if(log_p >= pi[k][stop][u]):
                    pi[k][stop][u] = log_p
        predicted_y_st = np.zeros((n), dtype = "int32")
        scores = np.log(np.zeros((n)))
        predicted_y_st[k] = stop
        k -= 1
        for v in range(m-2):
            b = lim
            if transition_params[v][stop][stop] != -np.inf:
                b = transition_params[v][stop][stop]
            log_p = pi[k+1][stop][v] + b 
            if (log_p>=scores[k]):
                scores[k] = log_p
                predicted_y_st[k] = v
        for k in range(len(words)-1, 0, -1):
            for v in range(m-2):
                b = lim
                if transition_params[v][predicted_y_st[k+1]][predicted_y_st[k+2]] != -np.inf:
                    b = transition_params[v][predicted_y_st[k+1]][predicted_y_st[k+2]]
                log_p = pi[k+1][predicted_y_st[k+1]][v] + b
                if (log_p>=scores[k]):
                    scores[k] = log_p
                    predicted_y_st[k] = v
        predicted_y[sentence_no] = list(predicted_y_st[1:-1])
    return predicted_y
def read_data_sentence_without_tag(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    words = {}
    labels = {}
    sentences = re.split('\n\n+', data)
    i = 0
    for sentence in sentences:
        words_st = []
        labels_st = []
        if sentence.strip() == "":
            continue
        lines = re.split('\n', sentence)
        for line in lines:
            word, label = line.split()
            words_st.append(word)
            labels_st.append(label)
        words[i] = words_st
        labels[i] = labels_st
        i += 1
    return words, labels
def estimate_emission_params_with_unk_words(words, labels, dictionary, k = 1):
    emission_counts = {} # A dictionary to store the count of each (x,y) pair
    label_counts = {} # A dictionary to store the count of each label y
    
    # Count the number of times each (x,y) pair occurs in the training set
    emission_counts["#UNK#"] = {}

    for i in range(len(words)):
        x = words[i]
        y = labels[i]
        
        if x not in dictionary:
            continue
        
        if x not in emission_counts:
            emission_counts[x] = {}
        
        if y not in emission_counts[x]:
            emission_counts[x][y] = 0
        emission_counts[x][y] += 1
        
        if y not in label_counts:
            label_counts[y] = 0
            emission_counts["#UNK#"][y] = 0

        label_counts[y] += 1
    
    # Calculate the emission probabilities using MLE
    emission_params = {}
    for x in emission_counts:
        emission_params[x] = {}
        for y in emission_counts[x]:
            if x != "#UNK#":
                emission_params[x][y] = emission_counts[x][y] / (label_counts[y] + k )
            else:
                emission_params[x][y] = k / (label_counts[y] + k )
    return emission_params
def change_emission_params_to_label(emission_params, label2idx):
    result = {}
    for x in emission_params:
        new_y_dict = {}
        y_dict = emission_params[x]
        for y in y_dict:
            new_y_dict[label2idx[y]] = y_dict[y]
        result[x] = new_y_dict
    return result

def eval_3(path, output_file):
    test_words_en_split, test_labels_en_split = read_data_sentence_without_tag(path + "dev.out")
    test_words_en, test_labels_en = read_data(path + "dev.out")
    train_words_en, train_labels_en = read_data_sentence(path + "train")
    label2idx_en, train_labels_idx_en, true_labels_idx_en = label2idx(train_labels_en, test_labels_en)
    transition_params_2step_en = estimate_transition_params_2step(train_labels_idx_en, label2idx_en)
    words_en, labels_en = read_data(path + "train")
    dict_en = np.unique(words_en)
    ep_en = estimate_emission_params_with_unk_words(words_en, labels_en, dict_en)
    emission_params_en = change_emission_params_to_label(ep_en, label2idx_en)
    pred_idx_vit_en_2 = viterbi_2(test_words_en_split, label2idx_en, emission_params_en, transition_params_2step_en)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, labels in pred_idx_vit_en_2.items():
            sentence = test_words_en_split[idx]
            for word, label in zip(sentence, labels):
                f.write("{} {}\n".format(word, list(label2idx_en.keys())[label]))
            f.write("\n")
eval_3("EN\\","EN\\dev.p3.out")
eval_3("FR\\","FR\\dev.p3.out")