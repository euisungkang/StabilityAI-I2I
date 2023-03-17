import io
import os
import warnings

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-wf0CC83Jm7Yw9XMCg7ZOeTiPrFVLykCcgIRJXXgXnJBVFGZu'

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    upscale_engine="esrgan-v1-x2plus",
    engine="stable-diffusion-768-v2-1"
)

answers = stability_api.generate(
    prompt="dystopian city black and white. people in the background, 1 girl in the foreground looking towards the crowd",
    #seed=892226758, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
    steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=7.5, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=768, # Generation width, defaults to 768 if not included.
    height=768, # Generation height, defaults to 768 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
)

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            global img
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ "-1-start.png") # Save our generated image with its seed number as the filename and the 1-start suffix so that we know this was our origin generation.

#upscaled = stability_api.upscale(
#    init_image=img
#)