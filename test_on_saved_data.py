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
import os
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

def get_mask_sentences(sentences,masked_indices,non_paded_sents,gt,vectorizer,words_used,mask_fraction=0.1):
    masked_sentences = []
    # batch_size, words, _ = masked_indices.size()
    # print(masked_indices.size())
    # IPython.embed()
    masked_indices = masked_indices.squeeze(-1)
    batch_size, words = masked_indices.size()
    index_value={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
    
    for i in range(batch_size):
        sentences = sentences[i]
        transformed = vectorizer.transform([sentences])
        # fully_indexed = []
        # IPython.embed()
        # for row in transformed:
            # fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})
        # sent_tf_idf_data = fully_indexed[0]
        
        # stop_words_idx = get_stop_words_idx(sentences)
        attention_weights_i = masked_indices[i]
        non_paded_attention = []; stop_words_attention = []
        # words_in_sent = sentences.split()
        words_in_sent = words_used[i]
        for j in range(words):
            if j >= len(words_in_sent):
                continue
            is_word = non_paded_sents[i][j][0]
            attention_val = attention_weights_i[j]
            # if words_in_sent[j] not in sent_tf_idf_data:
            #     updated_weight_mul = 1.0
            # else:
            #     updated_weight_mul = 1-sent_tf_idf_data[words_in_sent[j]] # doing this because we want less tfidf for important words
            if is_word:
                non_paded_attention.append([attention_val,j])
        
        non_paded_attention.sort(key = lambda x:x[0])
        ranked_masked_word_list = []
        for i_temp in non_paded_attention:
            mask_word_idx = i_temp[1]
            curr_word = words_in_sent[mask_word_idx]
            # if curr_word not in ranked_masked_word_list:
            ranked_masked_word_list.append(mask_word_idx)
        num_words_masked = int(mask_fraction * len(ranked_masked_word_list))
        ranked_masked_word_list = ranked_masked_word_list[:num_words_masked]
        updated_sent = ""
        for j in range(words):
            if j >= len(words_in_sent):
                continue
            word_at_j = words_in_sent[j]
            if j not in ranked_masked_word_list:
                word_at_j = words_in_sent[j]
            else:
                word_at_j = "[MASK]" 
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
def compute_masked_sentences(sentences, masking_network, sentence_transformer, device,mask_fraction,gt="p",vectorizer=None):
    # word_embeddings, words,_ = sentence_transformer(sentences, device)

    masked_sentences = []
    # counter = 0
    for sent in sentences:
        we_use, words_used, paded_words_idx = sentence_transformer([sent], device)
        # we_use = torch.stack([we],dim=0)
        we_use = torch.stack(we_use, dim=0).to(device)
        we_use = we_use.to(device)
        attention_weights = masking_network.get_masked_embeddings(we_use)
        masked_sentence = get_mask_sentences([sent],attention_weights,paded_words_idx,mask_fraction=mask_fraction,gt=gt,vectorizer=vectorizer,
                                             words_used=words_used)[0]
        # masked_sentence_c = collate(masked_sentence)
        masked_sentences.append(masked_sentence)

    return masked_sentences

def get_old_redacted_data(data_dir,sentence_transformer,mask_percentage:int):
    # data_dir = "/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"
    p_val,q_val,_, _ = getPqFromNpyData(data_dir+"/{}/sentOut.npy".format(mask_percentage))
    # IPython.embed()
    p_feat_sampled = sentence_transformer.model.encode(p_val)
    q_feat_sampled = sentence_transformer.model.encode(q_val)
    # p_feat_sampled, q_feat_sampled = normaliseData(p_feat_sampled, q_feat_sampled, norm='l1')
    return p_feat_sampled,q_feat_sampled

def getRandomWithReplacement(inp_feat):
    inp_ret = []
    numSamplesretinp = []
    for i in range(inp_feat.shape[0]):
        numSamplesretinp.append(i)
    idxsSampleToUse = random.choices(numSamplesretinp,k=inp_feat.shape[0])
    for idx_inp in idxsSampleToUse:
        inp_ret.append(inp_feat[idx_inp])
    return inp_ret,idxsSampleToUse

def bootstrap_data_get_idx(p_feat_data,q_feat_data,alphas):
    
    p_feat_sampled, btstrap_p_idx = getRandomWithReplacement(p_feat_data)
    q_feat_sampled, btstrap_q_idx = getRandomWithReplacement(q_feat_data)
    p_feat_sampled, q_feat_sampled = normaliseData(p_feat_sampled, q_feat_sampled, norm='l1')
    p_feat_sampled = np.asarray(p_feat_sampled)
    q_feat_sampled = np.asarray(q_feat_sampled)
    data_c = []
    for alpha in alphas :
        divergence_glove = divergence_Knn(p_feat_sampled, q_feat_sampled,alpha=alpha)
        data_c.append([alpha,divergence_glove])
    data_c = np.asarray(data_c)
    data_c_sorted = sorted(data_c, key=lambda x: x[0])
    eps = get_line_plots_using_kneedle(data_c_sorted,0.00005)
    return eps,0.00005


def bootstrap_data_get_idx_old(p_feat_data,q_feat_data,alphas):
    
    p_feat_sampled, btstrap_p_idx = getRandomWithReplacement(p_feat_data)
    q_feat_sampled, btstrap_q_idx = getRandomWithReplacement(q_feat_data)
    p_feat_sampled = np.asarray(p_feat_sampled)
    q_feat_sampled = np.asarray(q_feat_sampled)
    data_c = []
    for alpha in alphas :
        divergence_glove = divergence_Knn(p_feat_sampled, q_feat_sampled,alpha=alpha)
        data_c.append([alpha,divergence_glove])
    data_c = np.asarray(data_c)
    data_c_sorted = sorted(data_c, key=lambda x: x[0])
    eps = get_line_plots_using_kneedle(data_c_sorted,delta=0.0005)
    return eps

if __name__ == '__main__':
    debug = False
    data_dir = "/home/vaibhav/ML/redact_using_transformer"
    # MEDAL : 
    train_sent, train_gt = getinp_data("/home/vaibhav/ML/redact_using_transformer" + "/medalData/traincancer.txt", 
                                      "/home/vaibhav/ML/redact_using_transformer" + "/medalData/trainnoncancer.txt")
    
    p_val,q_val,_, _ = getPqFromNpyData("/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"+"/0/sentOut.npy")
    
    parent_dir = "/home/vaibhav/ML/density_estimation/redaction_using_density_estimation/"
    save_dir = parent_dir+"medal/sent_full" # dir to save sent_full sent
    diver_gence_dir = parent_dir+"medal/divergence_compare_data"
    checkpoint_path = 'model_divergence_medal_trained_sent_big_transformer.pth'
    data_dir_old = "/home/vaibhav/ML/bartexps/smartMaskingValidationMedalSepTrain_diff_length"
    
    p_train,q_train  = get_p_q(train_sent,train_gt)
    # REDIT :
    # train_sent, train_gt = getinp_data_csv("data/reddit_data_final/reddit_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/reddit_data_final/smart_masking_redit_suicide_new_data/0/sentOut.npy")
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # checkpoint_path = 'model_divergence_reddit_trained_sent.pth'
    # data_dir_old = "data/reddit_data_final/smart_masking_redit_suicide_new_data"
    # parent_dir = "/home/vaibhav/ML/density_estimation/redaction_using_density_estimation/"
    # save_dir = parent_dir+"reddit/sent" # dir to save
    # diver_gence_dir = parent_dir+"reddit/divergence_compare_data"
    
    # Amazon:
    # train_sent, train_gt = getinp_data_csv("data/amazon_new_dataset_final/amazon_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/amazon_new_dataset_final/smart_amazon_utility_redaction_new_data/0/sentOut.npy")
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # checkpoint_path = 'model_divergence_amazom_trained_sent.pth'
    # data_dir_old = "data/amazon_new_dataset_final/smart_amazon_utility_redaction_new_data"
    # parent_dir = "/home/vaibhav/ML/density_estimation/redaction_using_density_estimation/"
    # save_dir = parent_dir+"amazon/sent" # dir to save
    # diver_gence_dir = parent_dir+"amazon/divergence_compare_data"
    # Political dataset
    # train_sent, train_gt = getinp_data_csv("data/political_data_final/political_dataset_traindataset.csv")
    # p_val,q_val,_,_ = getPqFromNpyData("data/political_data_final/political_new_data_smart_redaction/0/sentOut.npy")
    
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    # random.shuffle(p_train); random.shuffle(q_train)
    # The code is shuffling the elements in the lists `p_train` and `q_train` randomly. This means
    # that the order of the elements in both lists will be rearranged in a random order.
    # p_train = p_train[:2500];q_train = q_train[:2500]
    # checkpoint_path = 'model_divergence_political_trained_sent.pth'
    # data_dir_old = "data/political_data_final/political_new_data_smart_redaction"
    # parent_dir = "/home/vaibhav/ML/density_estimation/redaction_using_density_estimation/"
    # save_dir = parent_dir+"political/sent" # dir to save
    # diver_gence_dir = parent_dir+"political/divergence_compare_data"
    # Device configuration
    device_id = 0  # Change this to the desired GPU ID (use -1 for CPU)
    device = torch.device("cuda:0")
    threshold = 0.5 # if probability is greateer than thereshold mask that word.
    # Initialize sentence transformer and model
    sentence_transformer = SentenceTransformerWrapper().to(device)
    # Initialize masking network
    # p_train,q_train  = get_p_q(train_sent,train_gt)
    sentence_transformer.train(train_sent,p_train,q_train)
    embedding_dim = sentence_transformer.model.get_sentence_embedding_dimension()  # Get embedding dimension from SentenceTransformer
    pooling_layer = sentence_transformer.model[1]
    normalize_layer = sentence_transformer.model[2]
    masking_network = MaskingNetwork(embedding_dim,thershold=threshold).to(device)
    # train_sents = []; train_sents.extend(p_train); train_sents.extend(q_train)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_sent)
    
    checkpoint = torch.load(checkpoint_path)
    masking_network.load_state_dict(checkpoint['model_state_dict'])
    
    divergence_glob = None
    alphas = [1.1,10,20,30,50]
    masking_network.eval()
    data_to_plot = []
    for mask_perc in range(10,31,10):
        mask_fraction = mask_perc/100.0
        print(f"Currently masking {mask_perc} fraction of words")
        
        if mask_perc == 100:
            data_to_plot.append([mask_perc,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            continue
        sent_save_dir = "{}/{}".format(save_dir,mask_perc)
        # os.makedirs(sent_save_dir,exist_ok=True)
        p_text_sampled = np.load("{}/{}".format(sent_save_dir,"p.npy"), allow_pickle=True)
        q_text_sampled = np.load("{}/{}".format(sent_save_dir,"q.npy"), allow_pickle=True)
        # IPython.embed()
        # np.save("{}/{}".format(sent_save_dir,"p.npy"),p_text_sampled_array)
        # np.save("{}/{}".format(sent_save_dir,"q.npy"),q_text_sampled_array)
        # IPython.embed()
        p_feat_new= sentence_transformer.model.encode(p_text_sampled)
        q_feat_new = sentence_transformer.model.encode(q_text_sampled)
        # p_feat_new, q_feat_new = normaliseData(p_feat_new, q_feat_new, norm='l1')
        p_feat_old, q_feat_old = get_old_redacted_data(data_dir_old,sentence_transformer,mask_perc)
        # divergence     = divergence_Knn(p_feat, q_feat, k=5, alpha=2)
        # divergence_old = divergence_Knn(p_feat_old,q_feat_old,alpha=2,k=5)
        eps_old = []; eps_new= []; delta_old = []; delta_new = []
        eps_old_paper = []
        for bootstap_idx in range(5):
            eps_old_tmp,delta_old_tmp = bootstrap_data_get_idx(p_feat_old,q_feat_old,alphas)
            eps_new_tmp,delta_new_tmp = bootstrap_data_get_idx(p_feat_new,q_feat_new,alphas)
            eps_old_paper_tmp = bootstrap_data_get_idx_old(p_feat_old,q_feat_old,alphas)
            print(f"{eps_old_tmp},{delta_old_tmp},{eps_new_tmp},{delta_new_tmp} Paper = {eps_old_paper_tmp}")
            eps_old.append(eps_old_tmp)
            eps_new.append(eps_new_tmp)
            delta_old.append(delta_old_tmp)
            delta_new.append(delta_new_tmp)
            eps_old_paper.append(eps_old_paper_tmp)
        data_to_plot.append([mask_perc,np.mean(eps_new),np.std(eps_new),np.mean(delta_new),np.std(delta_new),
                             np.mean(eps_old),np.std(eps_old),np.mean(delta_old),np.std(delta_old)])
    dataToPlot = np.asarray(data_to_plot)
    print(dataToPlot)
    # os.makedirs(diver_gence_dir,exist_ok=True)
    # np.save("{}/{}.npy".format(diver_gence_dir,"amazon_data"), dataToPlot)

