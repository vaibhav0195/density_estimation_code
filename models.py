import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
import numpy as np
import math
from kneed import  KneeLocator
from renyi_estimator import estimate_rynei_scipy
import IPython
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import losses, SentencesDataset,InputExample
import random
import string

class SentenceTransformerWrapper(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2',num_words=256):
        super(SentenceTransformerWrapper, self).__init__()
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.num_words = num_words
        self.cls_token = self.model.tokenizer.cls_token
        self.pad_token = self.model.tokenizer.pad_token
        self.sep_token = self.model.tokenizer.sep_token
        
    def get_chunks_of_sentences(self,sentence):
        chunk_size=100
        words = sentence.split()  # Assume words are space-separated
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        return_set = []
        for chunk in chunks:
            sent_tmp = ""
            for word in chunk:
                sent_tmp = sent_tmp+" "+word
            return_set.append(sent_tmp)
        return return_set
    
    def train(self,train_sentences,p_train,q_train):
        train_examples = []
        idx_to_use_p_train = [i for i in range(len(p_train))]; idx_to_use_q_train = [i for i in range(len(q_train))]
        # random.shuffle(idx_to_use_p_train)
        # random.shuffle(idx_to_use_q_train)
        # idx_to_use_q_train = idx_to_use_q_train[:2500];idx_to_use_p_train = idx_to_use_p_train[:2500]
        
        for idx,sentence_a in enumerate(p_train):
            sentence_a_chunk = self.get_chunks_of_sentences(sentence_a)
            for sentence_a_eg in sentence_a_chunk:
            # if idx in idx_to_use_p_train:
                pair_sent = random.choice(p_train)
                sentence_b_chunks = self.get_chunks_of_sentences(pair_sent)
                sentence_b_eg = random.choice(sentence_b_chunks)
                train_examples.append(InputExample(texts=[sentence_a_eg, sentence_b_eg],label=1.0))  # Pair with another sentence from class A
        for idx,sentence_a in enumerate(q_train):
            sentence_a_chunk = self.get_chunks_of_sentences(sentence_a)
            for sentence_a_eg in sentence_a_chunk:
            # if idx in idx_to_use_p_train:
                pair_sent = random.choice(q_train)
                sentence_b_chunks = self.get_chunks_of_sentences(pair_sent)
                sentence_b_eg = random.choice(sentence_b_chunks)
                train_examples.append(InputExample(texts=[sentence_a_eg, sentence_b_eg],label=1.0)) 
        # Generate negative pairs
        idx_to_use_p_train = [i for i in range(len(p_train))]; idx_to_use_q_train = [i for i in range(len(q_train))]
        # random.shuffle(idx_to_use_p_train)
        # random.shuffle(idx_to_use_q_train)
        # idx_to_use_q_train = idx_to_use_q_train[:2500];idx_to_use_p_train = idx_to_use_p_train[:2500]
        for idx,sentence_a in enumerate(p_train):
            sentence_a_chunk = self.get_chunks_of_sentences(sentence_a)
            for sentence_a_eg in sentence_a_chunk:
            # if idx in idx_to_use_p_train:
                pair_sent = random.choice(q_train)
                sentence_b_chunks = self.get_chunks_of_sentences(pair_sent)
                sentence_b_eg = random.choice(sentence_b_chunks)
                train_examples.append(InputExample(texts=[sentence_a_eg, sentence_b_eg],label=0.0))  # Pair with another sentence from class A
        
        for idx,sentence_a in enumerate(q_train):
            sentence_a_chunk = self.get_chunks_of_sentences(sentence_a)
            for sentence_a_eg in sentence_a_chunk:
            # if idx in idx_to_use_p_train:
                pair_sent = random.choice(p_train)
                sentence_b_chunks = self.get_chunks_of_sentences(pair_sent)
                sentence_b_eg = random.choice(sentence_b_chunks)
                train_examples.append(InputExample(texts=[sentence_a_eg, sentence_b_eg],label=0.0)) 
                
        self.model._first_module().tokenizer = self.model._first_module().tokenizer.train_new_from_iterator(train_sentences, 100000)
        self.model._first_module().auto_model.resize_token_embeddings(len(self.model._first_module().tokenizer))
        # # Resize the model's token embeddings
        # IPython.embed()
        train_dataset = SentencesDataset(train_examples, self.model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

        # # Define a loss function (for example, using cosine similarity)
        train_loss = losses.CosineSimilarityLoss(self.model)
        # # Fine-tune the model with your sentences
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=2500)
        self.tokenizer = self.model.tokenizer
        self.cls_token = self.model.tokenizer.cls_token
        self.pad_token = self.model.tokenizer.pad_token
        self.sep_token = self.model.tokenizer.sep_token
        # self.model.save_pretrained("medal_trained_trans")

    def forward(self, sentences, device):
        word_embeddings = []
        words_ret = []
        non_paded_sents = []
        chunk_size=100
        for i, sentence in enumerate(sentences):
            words = sentence.split()  # Assume words are space-separated
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
            
            # sentence_embeddings, sentence_words, sentence_mask = [], [], []
            word_list = []
            word_emb_list = []
            current_word = ''
            current_word_embs = []
            non_paded_sents_tmp = []
            for chunk in chunks:
                chunk_text = ' '.join(chunk)
                tokenized = self.model.tokenize([chunk_text])
                input_ids = tokenized['input_ids'].to(device)
                token_embeddings = self.model.encode([chunk_text], output_value='token_embeddings', convert_to_tensor=True, device=device)[0]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                for j, token in enumerate(tokens):
                    if len(current_word_embs) >= chunk_size:
                        break
                    if token.startswith(self.cls_token) or token.startswith(self.sep_token) or token == self.pad_token:  # Skip special tokens and padding
                        continue
                    if token.startswith('##'):
                        current_word += token[2:]
                        current_word_embs.append(token_embeddings[j])
                    else:
                        if current_word:
                            word_list.append(current_word)
                            word_emb_list.append(torch.mean(torch.stack(current_word_embs), dim=0))
                            non_paded_sents_tmp.append(torch.tensor(True))
                        current_word = token
                        current_word_embs = [token_embeddings[j]]
                if len(current_word_embs) < chunk_size:
                    if current_word:
                        word_list.append(current_word)
                        word_emb_list.append(torch.mean(torch.stack(current_word_embs), dim=0))
                        non_paded_sents_tmp.append(torch.tensor(True))

            words_ret.append(word_list)
            word_embeddings.append(word_emb_list)
            # IPython.embed()
            non_paded_sents.append(non_paded_sents_tmp)

        max_len = max([len(i) for i in words_ret])
        words_new = []
        non_paded_sents_new = []
        word_embeddings_new = []
        for i in range(len(words_ret)):
            words_in_sent = words_ret[i]
            padding_in_sent = non_paded_sents[i]
            embeddings_in_sent = word_embeddings[i]
            for j in range(max_len-len(words_in_sent)):
                words_in_sent.append("")
                padding_in_sent.append(torch.tensor(False))
                embeddings_in_sent.append(torch.zeros((self.embedding_dim), device=device).view(self.embedding_dim))
            words_new.append(words_in_sent)
            non_paded_sents_new.append(torch.stack(padding_in_sent).view((len(padding_in_sent),1)))
            word_embeddings_new.append(torch.stack(embeddings_in_sent))
        return word_embeddings_new, words_new, non_paded_sents_new
    
    def contextual_embeddings(self, sentences, device):

        word_embeddings = []
        words_in_sent = []
        non_paded_sents = []
        chunk_size=100
        for i, sentence in enumerate(sentences):
            words = sentence.split()  # Assume words are space-separated
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
            
            # sentence_embeddings, sentence_words, sentence_mask = [], [], []
            word_list = []
            word_emb_list = []
            current_word = ''
            current_word_embs = []
            non_paded_sents_tmp = []
            for chunk in chunks:
                chunk_text = ' '.join(chunk)
                tokenized = self.model.tokenize([chunk_text])
                input_ids = tokenized['input_ids'].to(device)
                token_embeddings = self.model.encode([chunk_text], output_value='token_embeddings', convert_to_tensor=True, device=device)[0]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                for j, token in enumerate(tokens):
                    if len(current_word_embs) >= chunk_size:
                        break
                    if token.startswith(self.cls_token) or token.startswith(self.sep_token) or token == self.pad_token:  # Skip special tokens and padding
                        continue
                    if token.startswith('##'):
                        current_word += token[2:]
                        current_word_embs.append(token_embeddings[j])
                    else:
                        if current_word:
                            word_list.append(current_word)
                            word_emb_list.append(torch.mean(torch.stack(current_word_embs), dim=0))
                            non_paded_sents_tmp.append(torch.tensor(True))
                        current_word = token
                        current_word_embs = [token_embeddings[j]]
                if len(current_word_embs) < chunk_size:
                    if current_word:
                        word_list.append(current_word)
                        word_emb_list.append(torch.mean(torch.stack(current_word_embs), dim=0))
                        non_paded_sents_tmp.append(torch.tensor(True))

            # for i in range(abs((len(word_list))-chunk_size)):
            #     word_list.append("")
            #     word_emb_list.append(torch.zeros((self.embedding_dim), device=device).view(self.embedding_dim))
            #     non_paded_sents_tmp.append(torch.tensor(False))
            # non_paded_sents_tmp_o = [i for i in non_paded_sents_tmp if i == True]
            # word_list_o = [i for i in word_list if i != ""]
            # IPython.embed()
            words_in_sent.append(word_list)
            word_embeddings.append(torch.stack(word_emb_list))
            # IPython.embed()
            non_paded_sents.append(torch.stack(non_paded_sents_tmp).view((len(non_paded_sents_tmp),1)))

        return word_embeddings, words_in_sent, non_paded_sents
    
class MaskingNetwork(nn.Module):
    def __init__(self, embedding_dim, thershold=0.5, hidden_dim=128):
        super(MaskingNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.drop_o1 = nn.Dropout(p=0.2)
        self.drop_o2 = nn.Dropout(p=0.2)
        self.drop_o3 = nn.Dropout(p=0.2)
        self.attention = nn.Linear(hidden_dim, 1)
        # self.classifier = nn.Linear(embedding_dim, 1)
        self.thershold = thershold
        self.temprature = torch.tensor(100,dtype=torch.float32,requires_grad=True)
        # self.top_k = 
    def forward(self, word_embeddings_P, non_paded_sents):
        word_embeddings_P.requires_grad_()
        x = self.drop_o1(torch.tanh(self.fc1(word_embeddings_P)))
        x = self.drop_o2(torch.tanh(self.fc2(x)))
        x = self.drop_o3(torch.tanh(self.fc3(x)))
        x = torch.sigmoid(self.attention(x))
        num_words = int(x.size(1)*0.1)
        topk_vals, _ = torch.topk(x, num_words, dim=1, largest=False)
        top_k_vals = topk_vals[:,-1,:]
        top_k_vals_uq = top_k_vals.unsqueeze(-1)
        x_new = torch.sigmoid(self.temprature *(x-top_k_vals_uq)) 
        updated_embeddings = torch.mean(word_embeddings_P * x_new,dim=1)
        # IPython.embed()
        return updated_embeddings,x_new

    def get_masked_embeddings(self,word_embeddings_P):
        word_embeddings_P.requires_grad_()
        x = self.drop_o1(torch.tanh(self.fc1(word_embeddings_P)))
        x = self.drop_o2(torch.tanh(self.fc2(x)))
        x = self.drop_o3(torch.tanh(self.fc3(x)))
        x = self.attention(x)
        # num_words = int(x.size(1)*0.3)
        # topk_vals, _ = torch.topk(x, num_words, dim=1, largest=False)
        # top_k_vals = topk_vals[:,-1,:]
        # top_k_vals_uq = top_k_vals.unsqueeze(-1)
        # x_new = torch.sigmoid(self.temprature *(top_k_vals_uq-x)) 
        # updated_embeddings = torch.mean(word_embeddings_P * x_new,dim=1)
        return torch.sigmoid(x)

class MaskingNetwork_doug_technique(nn.Module):
    def __init__(self, embedding_dim, threshold=0.5, hidden_dim=128, number_words=2000):
        super(MaskingNetwork_doug_technique, self).__init__()
        
        # The masking layers now start with the number of words (input size is number_words)
        self.masking_1 = nn.Linear(number_words, hidden_dim)  # Adjusted to go from word mapping size to hidden_dim
        self.masking_2 = nn.Linear(hidden_dim, hidden_dim)
        self.masking_3 = nn.Linear(hidden_dim, number_words)  # Back to number_words to output probabilities for each word
        
        # Dropout layers to help with regularization
        self.drop_o1 = nn.Dropout(p=0.2)
        self.drop_o2 = nn.Dropout(p=0.2)
        self.drop_o3 = nn.Dropout(p=0.2)
        
        # Sigmoid will be used to output probabilities
        self.sigmoid = nn.Sigmoid()
        
        self.threshold = threshold

    def forward(self, word_embeddings_P, id_tensor, non_paded_sents):
        """
        :param word_embeddings_P: A tensor of shape (batch_size, seq_len, embedding_dim) containing sentence embeddings.
        :param id_tensor: A tensor of shape (batch_size, number_words) which maps the words in the input.
        :param non_paded_sents: A tensor indicating which words are not padding.
        """
        word_embeddings_P.requires_grad_()

        # Forward pass through the masking layers with tanh activations
        x = torch.tanh(self.masking_1(id_tensor))
        x = self.drop_o1(x)  # Apply dropout for regularization
        x = torch.tanh(self.masking_2(x))
        x = self.drop_o2(x)
        x = torch.tanh(self.masking_3(x))
        
        # Output probabilities of redacting each word using sigmoid
        word_probs = self.sigmoid(x)

        # Generate masked embeddings using the attention weights from word_probs
        updated_attention_weights = self.get_masked_embeddings(word_embeddings_P, word_probs, non_paded_sents, threshold=self.threshold)
        
        # Apply the mask to the embeddings and average the masked embeddings
        updated_embeddings = torch.mean(word_embeddings_P * updated_attention_weights.unsqueeze(-1), dim=1)

        return updated_embeddings, word_probs

    def get_masked_embeddings(self, sentences_embs, attention_weights, non_paded_sents, threshold=0.8):
        """
        Returns masked embeddings based on attention weights, maintaining gradient flow.

        :param sentences_embs: A tensor of shape (batch_size, seq_len, embedding_dim) containing sentence embeddings.
        :param attention_weights: A tensor of shape (batch_size, seq_len, number_words) containing redaction probabilities.
        :param non_paded_sents: A tensor of shape (batch_size, seq_len, 1) indicating which words are actual words.
        :param threshold: Mask if Probability > threshold.
        :return: A tensor of shape (batch_size, seq_len) containing updated attention weights.
        """
        batch_size, seq_len = attention_weights.size()

        # Iterate over each sentence in the batch
        for i in range(batch_size):
            # Extract embeddings and attention weights for the current sentence
            sentence_embs = sentences_embs[i]
            attention_weights_i = attention_weights[i]
            non_paded_sents_i = non_paded_sents[i].squeeze()

            # Get the indices of words whose probability of redaction is above the threshold
            indices = torch.nonzero(attention_weights_i[non_paded_sents_i] > threshold).squeeze()

            # Set those attention weights to 0 (i.e., redact the word)
            attention_weights_i[indices] = 0.0

        return attention_weights

class RenyiDivergenceLoss(nn.Module):
    def __init__(self, alpha=1.0, k=5, mask_fraction=0.1, embedding_dim=None, alphas=[1.1, 10, 20, 30, 50]):
        super(RenyiDivergenceLoss, self).__init__()
        self.alpha = alpha
        self.k = k
        self.mask_fraction = mask_fraction
        self.embedding_dim = embedding_dim
        self.alphas = alphas
        self.delta = 0.00008

    def forward(self, sentence_embeddings_P, sentence_embeddings_Q):
        # print(sentence_embeddings_Q,sentence_embeddings_P)
        divergence_p = self.estimate_rynei(sentence_embeddings_P, sentence_embeddings_Q, 2)
        divergence_q = self.estimate_rynei(sentence_embeddings_Q, sentence_embeddings_P, 2)
        
        return max(divergence_p,divergence_q)

    def estimate_rynei(self, X, Y, alpha=2, k=3, rounding=None):
        d = X.shape[1]
        m = Y.shape[0]

        # Epsilon to prevent division by zero or log of non-positive numbers
        epsilon = 1e-20

        def get_knn_tree(data, point, k):
            distances = torch.cdist(data, point.unsqueeze(0))
            knn_distances, knn_indices = torch.topk(distances.squeeze(0), k=k, largest=False, dim=0)
            return knn_distances, knn_indices

        k_p = torch.zeros(m, device=X.device, dtype=X.dtype, )
        k_q = torch.zeros(m, device=X.device, dtype=X.dtype, )
        rho = torch.zeros(m, device=X.device, dtype=X.dtype, )
        nu = torch.zeros(m, device=X.device, dtype=X.dtype,  )

        for i in range(m):
            y = Y[i]
            dist, indices = get_knn_tree(Y, y, k + 1)
            dist = dist[1:]  # Skip the point itself
            indices = indices[1:]

            k_q[i] = indices.size(0)
            nu[i] = dist.max()  # max distance
            max_dist = dist.max()

            # Ensure max_dist is positive to avoid issues with log or division
            max_dist = max_dist if max_dist > epsilon else epsilon

            # Find points in X within the maximum distance
            indicies_x_point = torch.where(torch.cdist(X, y.unsqueeze(0)).squeeze(0) <= max_dist)[0]
            k_p[i] = indicies_x_point.size(0)
            rho[i] = max_dist

            # Ensure rho[i] is positive
            rho[i] = rho[i] if rho[i] > epsilon else epsilon

        # Add epsilon to prevent division by zero
        kp_sum = k_p.sum() + epsilon
        kq_sum = k_q.sum() + epsilon

        # Initialize the total divergence
        r = torch.zeros(1, device=X.device, dtype=X.dtype, requires_grad=True)

        for i in range(m):
            p_density = (k_p[i] / kp_sum) * (1 / (rho[i] ** d + epsilon))
            q_density = (k_q[i] / kq_sum) * (1 / (nu[i] ** d + epsilon))

            # Clamp ratio to avoid extreme values and prevent overflow
            p_density = torch.clamp(p_density,min=epsilon,max=1e10)
            q_density = torch.clamp(q_density,min=epsilon,max=1e10)
            ratio = p_density/q_density

            # Add to the total divergence
            r =r+ ratio * k_q[i] / kq_sum

        # Add epsilon to prevent log of zero
        value = (1 / (alpha - 1)) * torch.log(r)

        # Ensure the result is non-negative
        value = torch.max(torch.tensor(0.0, dtype=torch.float, device=X.device), value)
        # Check for NaN in value
        if torch.isnan(value):
            print("NaN detected in estimate_rynei computation")
            print(f"rho: {rho}")
            print(f"nu: {nu}")
            print(f"k_p: {k_p}")
            print(f"k_q: {k_q}")
            print(f"ratio: {ratio}")
            print(f"r: {r}")
        return value



def pairwise_distances(x, y):
    # Ensure x and y are tensors
    if isinstance(x, list):
        x = torch.stack(x)
    if isinstance(y, list):
        y = torch.stack(y)

    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    y_square = torch.sum(y ** 2, dim=1, keepdim=True).T
    xy = torch.matmul(x, y.T)
    return torch.sqrt(x_square + y_square - 2 * xy + 1e-8)


def knn_density_estimation(distances, k):
    # Ensure distances is a tensor
    if isinstance(distances, list):
        distances = torch.stack(distances)

    knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
    radius = knn_distances[:, -1]
    volume = (radius ** distances.size(1))
    density = k / (distances.size(0) * volume)
    return density

# def get_line_plots_using_kneedle(sorted_points,x,y,delta_diff,delta=0.1):
#     counter = 0
#     # for x_i,y_i in sorted_points:
#     #     if x_i == x and y_i==y:
#     #         break
#     #     counter +=1
#     # print(sorted_points)
#     x2 = sorted_points[-1][0];y2 = sorted_points[-1][1]+delta
#     x = sorted_points[-2][0];y = sorted_points[-2][1]+delta
#     slope, intercept = get_line_eqn(x, x2, y, y2)
#     # slope = 0.0
#     # intercept = y2 + 0.01
#     sqrt_term = slope * math.log(1/delta_diff)
#     eps = intercept + slope + 2*math.sqrt(sqrt_term)
#     plot_lines = []
#     for x, y in sorted_points:
#         plot_lines.append([x, slope * x + intercept])
#     return np.asarray(plot_lines), slope, intercept,eps

def get_line_eqn(x1,x2,y1,y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope,intercept

def get_line_plots_using_new_theorem(sorted_points,x1,y1,delta=0.1):
    counter = 0
    for x_i,y_i in sorted_points:
        if x_i == x1 and y_i==y1:
            break
        counter +=1
    x2 = sorted_points[counter+1][0];y2 = sorted_points[counter+1][1]+delta
    # slope, intercept = get_line_eqn(x, x2, y, y2)
    # x2 = sorted_points[-1][0];y2 = sorted_points[-1][1]
    # x1 = sorted_points[-2][0];y1 = sorted_points[-2][1]

    slope, intercept = get_line_eqn(x1, x2, y1, y2)
    eps = intercept + slope
    exp_term = -((-intercept-slope+eps)**2)/(4*slope)
    delta = math.exp(exp_term)
    min_term_1 = math.sqrt(math.pi*slope)
    common_deno = 1+(eps-intercept-slope)/2*slope
    min_term_2 = 1/common_deno
    min_term_3 = 1/(common_deno+math.sqrt((common_deno**2)+(4/(math.pi*slope))))
    min_term = min(min_term_1,min_term_2,min_term_3)
    delta = delta*min_term
    plot_lines = []
    for x, y in sorted_points:
        plot_lines.append([x, slope * x + intercept])
    return np.asarray(plot_lines), slope, intercept,eps,delta

def create_id_tensor(words,word_dicts, number_words):

    word_ids = []
    for sent_word in words:
        tmp_words = []
        for word in sent_word:
            tmp_words.append(word_dicts[word])
        word_ids.append(torch.tensor(tmp_words))
    word_ids = torch.stack(word_ids,dim=0)
    batch_size, seq_len = word_ids.size()

    # Initialize a tensor of zeros with shape (batch_size, seq_len, number_words)
    id_tensor = torch.zeros(batch_size, seq_len, number_words)

    # Scatter the 1s to the correct positions (one-hot encoding)
    id_tensor.scatter_(2, word_ids.unsqueeze(-1), 1.0)
    
    return id_tensor
