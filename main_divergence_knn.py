import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader, Dataset
from get_data import getinp_data,get_p_q,getinp_data_csv,getPqFromNpyData
from tqdm import tqdm
from models import *
import random
import torch.nn.functional as F
import IPython
from util import *
from sklearn.feature_extraction.text import TfidfVectorizer

class SentenceDataset(Dataset):
    def __init__(self, sentences,ground_truth):
        self.sentences = sentences
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx],self.ground_truth[idx]


def create_data_loader(sentences, ground_truth,batch_size=64):
    dataset = SentenceDataset(sentences,ground_truth)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def collate(review):
    updatedSent = ""
    prevWord = None
    for word in review.split():
        if prevWord is None:
            updateSentence = True
        else:
            if prevWord == word:
                updateSentence = False
            else:
                updateSentence = True
        if updateSentence:
            updatedSent = updatedSent + " " + word
            prevWord = word
    return updatedSent.strip()

def get_mask_sentences(sentences,masked_indices,non_paded_sents,gt,words_used,mask_fraction=0.1):
    masked_sentences = []
    # batch_size, words, _ = masked_indices.size()
    # print(masked_indices.size())
    # IPython.embed()
    masked_indices = masked_indices.squeeze(-1)
    batch_size, words = masked_indices.size()
    for i in range(batch_size):
        sentences = sentences[i]
        # stop_words_idx = get_stop_words_idx(sentences)
        attention_weights_i = masked_indices[i]
        non_paded_attention = []; stop_words_attention = []
        words_in_sent = words_used[i]
        for j in range(words):
            is_word = non_paded_sents[i][j][0]
            attention_val = attention_weights_i[j]
            if is_word:
                non_paded_attention.append([attention_val,j])
        
        non_paded_attention.sort(key = lambda x:x[0])
        # IPython.embed()
        ranked_masked_word_list = []
        for i_temp in non_paded_attention:
            mask_word_idx = i_temp[1]
            curr_word = words_in_sent[mask_word_idx]
            if curr_word not in ranked_masked_word_list:
                ranked_masked_word_list.append(curr_word)
        num_words_masked = int(mask_fraction * len(ranked_masked_word_list))
        ranked_masked_word_list = ranked_masked_word_list[:num_words_masked]
        updated_sent = ""
        for j in range(words):
            if j >= len(words_in_sent):
                continue
            word_at_j = words_in_sent[j]
            if word_at_j not in ranked_masked_word_list:
                word_at_j = words_in_sent[j]
            else:
                word_at_j = "maskgusaintcd" 
            updated_sent = updated_sent+" "+word_at_j
            updated_sent = collate(updated_sent)
        masked_sentences.append(updated_sent.strip())
    return masked_sentences

def convert_pq_to_data(p_text,q_text):
    val_sents = []; gts = [0 for _ in p_text]
    val_sents.extend(p_text)
    gts_q = [1 for _ in q_text]
    gts.extend(gts_q)
    val_sents.extend(q_text)
    return val_sents,gts

def estimate(X, Y, k=3,temperature=100):
    d = X.shape[1]  # Dimensionality of the input
    d = torch.tensor(d, dtype=torch.float32)
    temperature = torch.tensor(temperature,dtype=torch.float32)
    eps = torch.tensor(1e-10, dtype=torch.float32)
    d.requires_grad_()
    eps.requires_grad_()
    temperature.requires_grad_()
    def get_k_nearest_neighbors(X, Y, k):
        dists = torch.cdist(X, Y)  # Pairwise distances between X and Y
        topk_dists, _ = torch.topk(dists, k, largest=False)  # Get the top-k smallest distances
        return topk_dists
    
    rho_dists = get_k_nearest_neighbors(X, X, k + 1)
    rho = torch.max(rho_dists[:, 1:], dim=1)[0]
    rho = rho+ eps
    nu_dists = get_k_nearest_neighbors(X, Y, k)
    nu = torch.max(nu_dists, dim=1)[0]
    nu = nu+eps
    k_p = torch.sum(torch.sigmoid(temperature * (rho.unsqueeze(1) - rho_dists[:, 1:])), dim=1)
    k_q = torch.sum(torch.sigmoid(temperature * (nu.unsqueeze(1) - nu_dists)), dim=1)  
    kp_sum = k_p.sum() + eps
    kq_sum = k_q.sum() + eps
    logp_sub = d * torch.log(rho)
    logq_sub = d * torch.log(nu)
    logp = torch.log(k_p / kp_sum) - logp_sub
    logq = torch.log(k_q / kq_sum) - logq_sub
    log_sub = logp-logq
    divisiion = k_p/kp_sum
    mult = divisiion*log_sub
    r = torch.sum(mult)
    return r

# Compute masked sentences using the trained model
def compute_masked_sentences(sentences, masking_network, sentence_transformer, device,mask_fraction,gt="p"):
    # word_embeddings, words,_ = sentence_transformer(sentences, device)

    masked_sentences = []
    # counter = 0
    for sent in sentences:
        we_use, words_used, paded_words_idx = sentence_transformer([sent], device)
        # we_use = torch.stack([we],dim=0)
        we_use = torch.stack(we_use, dim=0).to(device)
        we_use = we_use.to(device)
        attention_weights = masking_network.get_masked_embeddings(we_use)
        masked_sentence = get_mask_sentences([sent],attention_weights,paded_words_idx,mask_fraction=mask_fraction,gt=gt,words_used=words_used)[0]
        # masked_sentence_c = collate(masked_sentence)
        masked_sentences.append(masked_sentence)

    return masked_sentences

def get_old_redacted_data(data_dir,sentence_transformer,mask_percentage:int):
    # data_dir = "/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"
    p_val,q_val,_, _ = getPqFromNpyData(data_dir+"/{}/sentOut.npy".format(mask_percentage))
    p_feat_sampled = sentence_transformer.model.encode(p_val)
    q_feat_sampled = sentence_transformer.model.encode(q_val)
    p_feat, q_feat = normaliseData(p_feat_sampled, q_feat_sampled, norm='l2')
    return p_feat,q_feat

if __name__ == '__main__':
    debug = False
    data_dir = "/home/vaibhav/ML/redact_using_transformer"
    # Device configuration
    device_id = 0  # Change this to the desired GPU ID (use -1 for CPU)
    device = torch.device("cuda:1")
    threshold = 0.5 # if probability is greateer than thereshold mask that word.
    # Initialize sentence transformer and model
    sentence_transformer = SentenceTransformerWrapper().to(device)
    
    # Medal Dataset
    # train_sent, train_gt = getinp_data_csv("data/reddit_data_final/reddit_traindataset.csv")
    train_sent, train_gt = getinp_data(data_dir + "/medalData/traincancer.txt", data_dir + "/medalData/trainnoncancer.txt")
    val_sent, val_gt = getinp_data(data_dir + "/medalData/cancerSentVal.txt", data_dir + "/medalData/noncancerSentVal.txt")
    p_val,q_val  = get_p_q(val_sent,val_gt)
    p_train,q_train  = get_p_q(train_sent,train_gt)
    p_val,q_val,_, _ = getPqFromNpyData("/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"+"/0/sentOut.npy")
    checkpoint_path = 'model_divergence_medal_trained_sent_big_transformertmp2.pth'
    data_dir_old = "/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"
    sentence_transformer.train(train_sent,p_train,q_train)
    # p_feat_old, q_feat_old = get_old_redacted_data(data_dir_old,sentence_transformer,10)
    ####################
    
    ##REDIT dataset
    # train_sent, train_gt = getinp_data_csv("data/reddit_data_final/reddit_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/reddit_data_final/smart_masking_redit_suicide_new_data/0/sentOut.npy")
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # checkpoint_path = 'model_divergence_reddit_trained_sent.pth'
    ##################
    
    ##Amazon dataset
    # train_sent, train_gt = getinp_data_csv("data/amazon_new_dataset_final/amazon_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/amazon_new_dataset_final/smart_amazon_utility_redaction_new_data/0/sentOut.npy")
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # checkpoint_path = 'model_divergence_amazom_trained_sent.pth'
    ##################
    
    ##Political dataset
    # train_sent, train_gt = getinp_data_csv("data/political_data_final/political_dataset_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/political_data_final/political_new_data_smart_redaction/0/sentOut.npy")
    # # IPython.embed()
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # checkpoint_path = 'model_divergence_political_trained_sent_new.pth'
    # # IPython.embed()
    # ####################
    # random.shuffle(p_train); random.shuffle(q_train)
    # p_train = p_train[:2500];q_train = q_train[:2500]
    # train_sent = []; train_sent.extend(p_train); train_sent.extend(q_train)
    # sentence_transformer.train(train_sent,p_train,q_train)
    
    # MEDAL :
    p_feat_old, q_feat_old = get_old_redacted_data(data_dir_old,sentence_transformer,10)
    #REDIT : 
    # p_feat_old, q_feat_old = get_old_redacted_data("data/reddit_data_final/smart_masking_redit_suicide_new_data",sentence_transformer,10)
    # #AMAZON:
    # p_feat_old, q_feat_old = get_old_redacted_data("data/amazon_new_dataset_final/smart_amazon_utility_redaction_new_data",sentence_transformer,10)
    # POLITICAL:
    # p_feat_old, q_feat_old = get_old_redacted_data("data/political_data_final/political_new_data_smart_redaction",sentence_transformer,10)
    
    batch_size = 64
    p_gt = [0 for _ in p_train]; q_gt = [1 for _ in q_train]
    data_loader_P = create_data_loader(p_train, p_gt, batch_size); data_loader_Q = create_data_loader(q_train, q_gt, batch_size)
    # Initialize masking network
    embedding_dim = sentence_transformer.model.get_sentence_embedding_dimension()  # Get embedding dimension from SentenceTransformer
    # pooling_layer = sentence_transformer.model[1]
    # normalize_layer = sentence_transformer.model[2]
    masking_network = MaskingNetwork(embedding_dim,thershold=threshold).to(device)
    # train_sents = []; train_sent.extend(p_train); train_sent.extend(q_train)
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(train_sent)
    optimizer = optim.Adam(masking_network.parameters(), lr=0.001)
    num_epochs =5
    # IPython.embed()
    best_val_acc = None
    
    divergence_old = divergence_Knn(p_feat_old,q_feat_old,alpha=2,k=5)
    del p_feat_old; del q_feat_old
    divergence_glob = None
    losses_to_print = []
    for epoch in range(num_epochs):  # Number of epochs
        masking_network.train()  # Set model to training mode
        total_loss = 0.0
        loss_tmps = []
        counter = 0
        for batch_P, batch_Q in tqdm(zip(data_loader_P, data_loader_Q)):
            sentences_P_batch = batch_P[0]
            sentences_Q_batch = batch_Q[0]
            # IPython.embed()
            # batch_sents = sent
            input_batch = []; input_batch.extend(list(sentences_P_batch))
            input_batch.extend(list(sentences_Q_batch))
            # P, P_words,non_paded_sents_p = sentence_transformer(sentences_P_batch, device)
            # Q, Q_words,non_paded_sents_q = sentence_transformer(sentences_Q_batch, device)
            # sents = [];sents.extend(P);sents.extend(Q)
            # padding = [];padding.extend(non_paded_sents_p);padding.extend(non_paded_sents_q)
            # sents = torch.stack(sents,dim=0).to(device)
            # padding = torch.stack(padding,dim=0).to(device)
            sents,_,padding = sentence_transformer(input_batch, device)
            # IPython.embed()
            sents = torch.stack(sents,dim=0).to(device)
            padding = torch.stack(padding,dim=0).to(device)
            # IPython.embed()
            optimizer.zero_grad()  # Zero the gradients
            updated_embeddings,_ = masking_network(sents,padding)
            sent_p = updated_embeddings[:len(sentences_P_batch)]; sent_q = updated_embeddings[len(sentences_P_batch):]
            loss_1 = estimate(sent_p, sent_q, k=3); loss_2 = estimate(sent_q, sent_p, k=3)
            
            default_val = torch.tensor(1e-2,dtype=torch.float32)
            default_val.requires_grad_()
            default_val.to(device)
            loss = torch.max(loss_1,loss_2)
            loss_fin = torch.max(loss,default_val)
            # print(loss_fin.item())
            loss_tmps.append(loss_fin.item())
            counter+=1
            if counter%10==0:
                losses_to_print.append(sum(loss_tmps)/10.0)
                loss_tmps = []
                # print(losses_to_print)
            # IPython.embed()
            # IPython.embed()
            # get_dot = register_hooks(loss_fin)
            loss_fin.backward()  # Backpropagation: Compute gradients
            # dot = get_dot()
            # dot.save('tmp.dot')
            # IPython.embed()
            optimizer.step()  # Update model parameters
            total_loss += loss_fin.item() * len(sentences_P_batch)
        # losses_to_print.append(total_loss / (batch_size*len(data_loader_P)))
        print(losses_to_print)
        print(f"Epoch {epoch}, Average Loss: {total_loss / (batch_size*len(data_loader_P))}")
        mask_fraction = 0.1
        # print(f"Doing Validation for mask percentage {mask_fraction}")
        # masking_network.eval()
        
        p_text_sampled = compute_masked_sentences(p_val, masking_network, sentence_transformer,
                                                                    device,gt="p",mask_fraction=mask_fraction)
        q_text_sampled = compute_masked_sentences(q_val, masking_network, sentence_transformer,
                                                                    device,gt="q",mask_fraction=mask_fraction)
        p_feat_sampled = sentence_transformer.model.encode(p_text_sampled)
        q_feat_sampled = sentence_transformer.model.encode(q_text_sampled)
        p_feat, q_feat = normaliseData(p_feat_sampled, q_feat_sampled, norm='l2')
        divergence     = divergence_Knn(p_feat, q_feat, k=5, alpha=2)
        
        print(f" ****Epoch={epoch}, Divergence={divergence} Old_divergence= {divergence_old}*****")
        if divergence_glob is None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': masking_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            divergence_glob = divergence
        else:
            if divergence_glob >= divergence:
                torch.save({
                'epoch': epoch,
                'model_state_dict': masking_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                }, checkpoint_path)
                divergence_glob = divergence
    # print(losses_to_print)
