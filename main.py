
from gramformer import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector


def correct(influent_sentence):
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    cadena_resultado = next(iter(corrected_sentences))
    return cadena_resultado


def main_correction():
   

print(correct("How is they?"))