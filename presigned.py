# import os
# from minio import Minio
# from datetime import timedelta



# # MinIO config
# s3_client = Minio(
#     "moja.shramba.arnes.si",
#     access_key=os.getenv("S3_ACCESS_KEY"),
#     secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
#     secure=True
# )
# bucket_name = "zrsvn-rag-najdbe"

# url = s3_client.presigned_get_object(
#     bucket_name,
#     "04_Mehkuzci/A_1_2_Volceke_Unio_crassus_CKFF.pdf",
#     # "08_Metulji/Coenonympha_oedippus_Slovenska Istra_poročilo-december2020.pdf",
#     expires=timedelta(hours=1) # Use 7 days (604800 seconds)
# )

# print(url)

import os
from minio import Minio
from datetime import timedelta
# Za branje ključev iz datoteke .env
from dotenv import load_dotenv 

# 1. Naložimo ključe iz datoteke .env
load_dotenv()


# MinIO config
s3_client = Minio(
    "moja.shramba.arnes.si",
    access_key=os.getenv("S3_ACCESS_KEY"),
    secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    secure=True
)
bucket_name = "zrsvn-rag-najdbe-vecji"

# Klic je sedaj izveden z veljavnimi ključi.
url = s3_client.presigned_get_object(
    bucket_name,
    # "04_Mehkuzci/A_1_2_Volceke_Unio_crassus_CKFF.pdf",
    "08_Metulji/Phengaris_nausithous_Ličenca_Dravinja_december_2020.pdf",
    expires=timedelta(hours=1) # Uporabimo 7 dni (604800 sekund)
)

print(url)

# import subprocess
# import shlex

# def generate_presigned_url(mc_alias, bucket, object_path, expiry="01h00m00s"):
#     cmd = f'mc share download --expire {expiry} "{mc_alias}/{bucket}/{object_path}"'
#     process = subprocess.run(
#         shlex.split(cmd),
#         capture_output=True,
#         text=True
#     )

#     if process.returncode != 0:
#         raise RuntimeError(f"mc failed: {process.stderr}")

#     # Parse the output lines
#     lines = process.stdout.strip().splitlines()
#     for line in lines:
#         if line.startswith("Share:"):
#             return line.replace("Share: ", "").strip()

#     raise RuntimeError("Presigned URL not found in mc output")


# # Example usage
# if __name__ == "__main__":
#     url = generate_presigned_url(
#         mc_alias="myminio",
#         bucket="zrsvn-rag-najdbe-vecji",
#         object_path="08_Metulji/Coenonympha_oedippus_Slovenska Istra_poročilo-december2020.pdf",
#         expiry="01h00m00s"
#     )
#     print(url)
