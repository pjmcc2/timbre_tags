# For calculating representation metrics

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff
import ot
from sklearn.metrics import ndcg_score
import torch


def cosine_sim(A,B,upper=False,numpy=False):
    """
    Given Matrices of size A = [num_embs_A,emb_dim] and B = [num_embs_B,emb_dim] where num_embs can be different
    returns a matrix of size [num_embs_A,num_embs_B] where each element is the cosine similarity of each row between A and B
    """
    if len(A.shape) > 2 or len(B.shape) > 2:
        raise ValueError(f"Too many dimensions. Expected 2 but found: {A.shape} and {B.shape}.")
    A = torch.tensor(A) if not isinstance(A, torch.Tensor) else A
    B = torch.tensor(B) if not isinstance(B, torch.Tensor) else B
    A_norm = A / A.norm(2,dim=1, keepdim=True)  # Normalizing each row of A
    B_norm = B / B.norm(2,dim=1, keepdim=True) # Normalizing each row of B

    cs = torch.mm(A_norm, B_norm.T)
    if upper:
      cs=cs.triu()
    if  numpy:
      cs = cs.numpy()
    return cs



def elementwise_cosine_sim(A,B):
  if len(A.shape) > 2 or len(B.shape) > 2:
        raise ValueError(f"Too many dimensions. Expected 2 but found: {A.shape} and {B.shape}.")
  A = torch.tensor(A) if not isinstance(A, torch.Tensor) else A
  B = torch.tensor(B) if not isinstance(B, torch.Tensor) else B
  A_norm = A / A.norm(2,dim=1, keepdim=True)  # Normalizing each row of A
  B_norm = B / B.norm(2,dim=1, keepdim=True)  # Normalizing each row of B
  return torch.sum(A_norm * B_norm,axis=1)

def sample_points(A, B, n_samples, sample_size): # AI-code
    nA = A.size(0)
    nB = B.size(0)

    idx_A = torch.randint(0, nA, (n_samples, sample_size), device=A.device)
    idx_B = torch.randint(0, nB, (n_samples, sample_size), device=B.device)

    A_batch = A[idx_A]   # (n_samples, sample_size, d)
    B_batch = B[idx_B]   # (n_samples, sample_size, d)

    return A_batch, B_batch

def chamfer_distance_batched(A, B): # AI-Code
    """
    A: (B, N, D)
    B: (B, M, D)
    Returns: (B,) Chamfer distances
    """

    # Compute squared norms
    A_sq = (A ** 2).sum(dim=2, keepdim=True)        # (B, N, 1)
    B_sq = (B ** 2).sum(dim=2).unsqueeze(1)         # (B, 1, M)

    # Pairwise squared distances: (B, N, M)
    dist = A_sq + B_sq - 2 * (A @ B.transpose(1, 2))

    # Nearest neighbor distances
    dist_A_to_B = dist.min(dim=2).values            # (B, N)
    dist_B_to_A = dist.min(dim=1).values            # (B, M)

    # Mean over points, sum the two directions
    cd = dist_A_to_B.mean(dim=1) + dist_B_to_A.mean(dim=1)

    return cd

def get_chamfer_distribution_batched(A, B, n_samples=100, sample_size=1000): # AI-code
    A_batch, B_batch = sample_points(A, B, n_samples, sample_size)
    return chamfer_distance_batched(A_batch, B_batch)



#AI Code
def hausdorff(A,B):

  # Calculate directed Hausdorff distance from A to B
  d_A_to_B = directed_hausdorff(A, B)[0]

  # Calculate directed Hausdorff distance from B to A
  d_B_to_A = directed_hausdorff(B, A)[0]

  # The full (symmetric) Hausdorff distance is the maximum of the two
  # hausdorff_distance = max(d_A_to_B, d_B_to_A)
  return max(d_A_to_B, d_B_to_A)

def get_OT_l2loss(A,B):
  res = ot.solve_sample(A,B)
  return res.value_linear

def get_mean_dist(A,B):
  mean_vec = np.mean(A-B,axis=0)
  return np.sqrt(np.linalg.vector_norm(mean_vec,axis=0))



def get_lin_sep(X_1,X_2,reg_strength=1.0):
  """
  Returns the 10-fold cross_val accuracy of a support vector classifier on the two embeddings.
  """
  Y = np.array([0 if i < len(X_1) else 1 for i in range(len(X_1)+len(X_2))])
  model = SVC(kernel='linear',C=reg_strength,random_state=42) # default is 1
  score = cross_val_score(model,np.vstack([X_1,X_2]),Y,scoring="accuracy",cv=KFold(n_splits=10,shuffle=True,random_state=42))
  return score.mean()


def calc_dist_metrics(A,B,*args,**kwargs):
  res = {}
  

  res["linear_sep"] = get_lin_sep(A,B) # TODO add kwargs


  res["mean_dist" ] = get_mean_dist(A,B)


  res["hausdorff_dist"] = hausdorff(A,B)


  res["ot_loss"] = get_OT_l2loss(A,B)

  
  return res



def get_consistency_torch(A, B): # AI-improved code
    """
    A, B: (N, D) torch tensors
    Returns: scalar consistency score
    """
    diff = A - B
    frob = torch.norm(diff, p='fro')      # Frobenius norm
    return 1 - frob / A.size(0)


def get_pair_alignment(A,B):
  return np.mean(elementwise_cosine_sim(A,B).numpy())

def get_ndcg(A, B):
    A_masked = torch.relu(A)
    return ndcg_score(A_masked.cpu().numpy(), B.cpu().numpy())


def calc_rep_metrics(A, A2, B, supervised=False): # AI-improved code
    res = {}

    # --- Self similarities ---
    orig_self = cosine_sim(A, A,upper=True)         # text-text
    trans_self = cosine_sim(A2, A2,upper=True)       # audio-audio

    #res["self_ndcg"] = ndcg_score(orig_self + 1, trans_self)
    res["self_ndcg"] = get_ndcg(orig_self, trans_self)
    res["self_consistency"] = get_consistency(orig_self, trans_self)

    # free memory early
    del trans_self

    # --- Text vs transformed text ---
    orig_trans = cosine_sim(A, A2)

    #res["pseudo_ndcg"] = ndcg_score(orig_self + 1, orig_trans)
    res["pseudo_ndcg"] = get_ndcg(orig_self, orig_trans)
    res["pseudo_consistency"] = get_consistency(orig_self, orig_trans)

    del orig_self
    del orig_trans

    # --- Cross similarities ---
    orig_cross = cosine_sim(A, B)         # text-audio
    trans_cross = cosine_sim(A2, B)       # audio-audio

    #res["cross_ndcg"] = ndcg_score(orig_cross + 1, trans_cross)
    res["cross_ndcg"] = get_ndcg(orig_cross, trans_cross)
    res["cross_consistency"] = get_consistency(orig_cross, trans_cross)

    del orig_cross
    del trans_cross

    # --- Optional supervised metric ---
    res["pair_alignment"] = get_pair_alignment(A, B) if supervised else -100

    return res



