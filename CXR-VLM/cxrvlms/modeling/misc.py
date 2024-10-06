import torch
import numpy as np
import random
import os


"""
Seeds for reproducibility.
"""


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


"""
Download FLAIR pre-trained weights.
"""


def wget_gdrive_secure(fileid, input_dir, filename):

    os.system("wget --header='Host: drive.usercontent.google.com' --header='User-Agent: Mozilla/5.0 (Windows NT 10.0;"
              " Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' --header='Accept: "
              "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
              "application/signed-exchange;v=b3;q=0.7' --header='Accept-Language: es-ES,es;q=0.9,en;q=0.8,fr;q=0.7' "
              "--header='Cookie: SEARCH_SAMESITE=CgQI2JkB; HSID=ACiPxzgpfNw883hR6; SSID=AsdboVzlxS6_ngiCu;"
              " APISID=fKvJNLo9BZTPS-t2/AVwsXYINesblDG67h; SAPISID=n8O2VGr9_bc0teEd/AWySLRfrvGpBVQwqS; __Secure-"
              "1PAPISID=n8O2VGr9_bc0teEd/AWySLRfrvGpBVQwqS; __Secure-3PAPISID=n8O2VGr9_bc0teEd/AWySLRfrvGpBVQwqS;"
              " SID=egjVQyBYHvnAfwobz5xta1GxwqpxVtJfZJ0WkpV2Fhl-VX_WjPy5fGiEOIjPafWPo10Dfw.; __Secure-1PSID=egjVQyBYH"
              "vnAfwobz5xta1GxwqpxVtJfZJ0WkpV2Fhl-VX_W3jCSlXT5npGDEddYRd0zRA.; __Secure-3PSID=egjVQyBYHvnAfwobz5xta1G"
              "xwqpxVtJfZJ0WkpV2Fhl-VX_WWWQAUUwNIjzNE1U1eANL9A.; AEC=Ae3NU9Og8VA7nJJCI3qxwfxhuA4e4neEr9n_oCVHi6DZKei9"
              "ESpUThM6Bw; NID=511=lnMj9MFUz5Zl03Cdos2E9wfiHhEZjKPgMOXjzfVHZuyeDXbgSPi_reqLbKTvWCynYPvPlqfG7OPzfN31t3G"
              "O0dHEl-Rpa940OtuoEex7yBMQ_GvPc9kiZG0rDe2oQRugzCz4TUpoJ0a36uxoTieSQ4feXCaQkqc2Su7xIhUAbhBNoVxnDvK65JInuF"
              "yJITR9B4vtUaf5LuAGsEC9_AX4_sRTovPOZSiY5eHMtl16zSGP3BmLOWBGkebe09i7C8HhajDWrUJRfPG6ECdiLbMJ9E5jarDEP2B9z"
              "LIDNSfHWpNdAqK_Vlq9pKXN1AvTLe7amiw5OyoEGb2LIrP9fiJwKLcLWIJz4w_Mf_bXjJy4NNngMvSsf16ea5y249o6aPiAex7MeQ3y"
              "VUKYIwq49EKSjghp45waIWqefWOSnlSVk0v5XlkfgtEcyp9YtjRo6TICmf2tf7WfvfoK; 1P_JAR=2024-01-14-23; __Secure-1P"
              "SIDTS=sidts-CjEBPVxjSssKHXZqzgA2s0uR857AVdkBlqxCIHN9TAyMoSjMhIUcB5neOQcAgxi5D1PPEAA; __Secure-3PSIDTS=s"
              "idts-CjEBPVxjSssKHXZqzgA2s0uR857AVdkBlqxCIHN9TAyMoSjMhIUcB5neOQcAgxi5D1PPEAA; SIDCC=ABTWhQHCpCYzm7AesDI"
              "uggin8j6y6UAsg_tPRuhpXjUqozr-KOaOcyif4gDNvUqWnfSZVwQjlJA; __Secure-1PSIDCC=ABTWhQFBknIAFMOKbR73TdpyjOh1"
              "VuixvvSwpRzsiSNyp0z6CL5enbfnNppJI0s0Om4Bu-CWSbE; __Secure-3PSIDCC=ABTWhQGBxChxCtw3BGzi-VyXFnUXFisrlgaDn"
              "9I1-ZV9KTZiC6-umgnSmcxZdvfKcf_vqACI90E' --header='Connection: keep-alive' 'https://drive.usercontent.go"
              "ogle.com/download?id=$fileid&export=download&authuser=0&confirm=t&uuid=3c6a12"
              "0c-625e-479d-8002-e33b584e4279&at=APZUnTX0w9kZJ5p8GzTjPDnO3L2s:1705276754199' -c -O '$pathid'".
              replace("$fileid", fileid).replace("$pathid", input_dir + filename))