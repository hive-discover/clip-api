from io import BytesIO
import math
import os, time
import random
import base58
from datetime import datetime
from opensearchpy import OpenSearch
import requests
from brisque import score_image
import numpy as np
from PIL import Image
from multiprocessing.pool import ThreadPool
import math
import weaviate

OPENSEARCH_HOSTS = os.environ.get("OPENSEARCH_HOSTS", None).split(",")
os_client = OpenSearch(OPENSEARCH_HOSTS)

weaviate_client = weaviate.Client(os.environ.get("WEAVIATE_HOST", None), additional_headers={"Authorization" : os.environ.get("WEAVIATE_AUTH", None)})

IMAGE_HOSTER_PREFIX = os.environ.get("IMAGE_HOSTER_PREFIX", "https://images.hive.blog/p")
CLIP_API_ADDRESS = os.environ.get("CLIP_API_ADDRESS", "http://127.0.0.1:8080")
CLIP_WORKER_HEARBEAT_URL = os.environ.get("CLIP_WORKER_HEARBEAT_URL", None)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))
NUM_IMAGE_WORKERS = int(os.environ.get("NUM_IMAGE_WORKERS", 3))

TOTAL_POSTS_FOUND = 0

def get_batch_query() -> dict:
    global TOTAL_POSTS_FOUND
    begin_from = random.randint(0, TOTAL_POSTS_FOUND - BATCH_SIZE) if TOTAL_POSTS_FOUND >= BATCH_SIZE else 0
    return {
        "size" : str(BATCH_SIZE),
        "from" : str(begin_from),
        "query" : {
            "bool" : {
                "must" : [
                    {"nested" : {
                        "path" : "jobs",
                        "query" : {
                            "bool" : {     
                                # Lang has to be calculated
                                "must_not" : [
                                    {"term" : {"jobs.imgs_described" : True}},                          
                                ],                
                            }
                        }
                    }}
                ]
            }
        },
        "_source" : {
            "includes" : ["timestamp", "image"]
        },
        "sort" : [
            {"timestamp" : {"order" : "desc"}}
        ]
    }

def get_next_batch() -> list:
    global TOTAL_POSTS_FOUND
    res = os_client.search(index="hive-posts", body=get_batch_query())
    TOTAL_POSTS_FOUND = res["hits"]["total"]["value"]
    return res["hits"]["hits"]

def download_image(url : str) -> Image:
    # Get image from Hive Image Service
    if not url.startswith(IMAGE_HOSTER_PREFIX):
        img_hash = base58.b58encode(url.encode("utf-8")).decode("utf-8")
        url = f"{IMAGE_HOSTER_PREFIX}/{img_hash}"

    url = url if "?" in url else url + "?a=b"
    if "format=jpeg" not in url:
        url += "&format=jpeg"

    # Specify width and height to 1024 to get a smaller image ==> faster to process (Image Service only downscales)
    url += "&width=1024&height=1024&mode=fit"

    try:
        return Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        print(f"Error while downloading image ('{url}'): {e}")
        return None

def describe_image(img : Image) -> list:
    # Send image to CLIP API
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resp = requests.post(f"{CLIP_API_ADDRESS}/encode-image-file", files={"file" : img_bytes})
    if resp.status_code == 200:
        return resp.json()

    # Do not catch an error to stop this execution and throw an Error when something went wrong
    raise Exception(f"CLIP-API returned unhealthy status-code: {resp.status_code}")

def duplicate_img_exists(embedding : list, threshold : float = 1.9) -> list:
    query = {
        "size" : 10,
        "query" : {
            "script_score": {
                "query": { "match_all": {} },
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "clip_vector",
                        "query_value": embedding,
                        "space_type": "cosinesimil"
                    }
                }
            }     
        },
        "_source" : {
            "includes" : ["image_hash", "brisque_score"]
        }    
    }

    # Do search and return docs if there sim-score is above threshold
    res = os_client.search(index="hive-imgs", body=query)
    hits = res["hits"]["hits"]
    return [hit for hit in hits if hit["_score"] > threshold]

def hash_already_exists(url : str) -> bool:
    # Get image from Hive Image Service
    img_hash = base58.b58encode(url.encode("utf-8")).decode("utf-8")

    query = {
        "size" : 1,
        "query" : {
            "bool" : {
                "must" : [
                    {"term" : {"image_hash" : img_hash}}
                ]
            }
        },
        "_source" : {
            "includes" : ["nothing"]
        }
    }

    res = os_client.search(index="hive-imgs", body=query)
    return res["hits"]["total"]["value"] > 0

def process_image(img_url : str, timestamp : datetime, fails : int = 0) -> list:
    img_hash = base58.b58encode(img_url.encode("utf-8")).decode("utf-8")
    if hash_already_exists(img_url):
        return []

    img = download_image(img_url)
    if img is None:
        return []

    # Get embedding and brisque-score
    try:
        embedding = describe_image(img)
    except Exception as e:
        print(f"Error while describing image ('{img_url}'): {e}")
        return []

    try:
        brisque_score = score_image(np.array(img))
    except Exception as e:
        print(f"Error while calculating brisque-score ('{img_url}'): {e}")
        brisque_score = 1000 # do not filter out images with unknown brisque-score

    if math.isnan(brisque_score):
        brisque_score = 1000

    # Check for dups
    duplicates = duplicate_img_exists(embedding)
    if len(duplicates) > 0:
        # Add this hash to the duplicates
        update_bulk = []
        for dup in duplicates:
            prev_brisque_score = float(dup["_source"]["brisque_score"] if "brisque_score" in dup["_source"] else 1000)
            brisque_score = min(brisque_score, prev_brisque_score) if not math.isnan(prev_brisque_score) else brisque_score

            update_bulk.append({"update" : {"_index" : dup["_index"], "_id" : dup["_id"]}})
            update_bulk.append({"doc" : {"image_hash" : list(set(dup["_source"]["image_hash"] + [img_hash])), "brisque_score" : brisque_score}})
        return update_bulk

    # Add new image to Weaviate
    weave_uuid = weaviate_client.data_object.create(
        {"image_hash" : [img_hash], "brisque_score" : brisque_score},
        "HivePostsImageVectors",
        vector=embedding
    )

    # Add new image to ES
    return [{
        "index" : {
            "_index" : "hive-imgs",
            "_id" : img_hash
        }
    }, {
        "image" : img_url,
        "clip_vector" : embedding,
        "image_hash" : [img_hash],
        "brisque_score" : brisque_score,
        "timestamp" : timestamp.isoformat()
    }]


def combine_post_imgs(image_urls : list, post_id : str, post_timestamp : datetime) -> list:
    '''
        Get all images from the post and calc the average embedding
    '''
    # Hash urls and get clip-vectors and brise-score
    img_hashes = [base58.b58encode(url.encode("utf-8")).decode("utf-8") for url in image_urls]
    os_results = os_client.search(index="hive-imgs", body={
        "size" : len(image_urls),
        "query" : {
            "terms" : {"image_hash" : img_hashes}
        },
        "_source" : {
            "includes" : ["clip_vector", "brisque_score"]
        }
    })

    # Calc average embedding / brisque-score
    avg_embedding = np.zeros(512)
    avg_brisque_score = 1000
    for hit in os_results["hits"]["hits"]:
        if not hit["_source"]["clip_vector"] or len(hit["_source"]["clip_vector"]) != 512:
            continue

        if np.count_nonzero(avg_embedding) == 0:
            avg_embedding = np.array(hit["_source"]["clip_vector"])
        else:
            avg_embedding = np.mean([avg_embedding, hit["_source"]["clip_vector"]], axis=0) 

        if not hit["_source"]["brisque_score"] or hit["_source"]["brisque_score"] > 150:
            continue
        
        if avg_brisque_score == 1000:
            avg_brisque_score = hit["_source"]["brisque_score"]
        else:
            avg_brisque_score = np.mean([avg_brisque_score, hit["_source"]["brisque_score"]]) 

    if np.count_nonzero(avg_embedding) == 0:
        return []

    # Update post and add avg-clip-embedding and avg-brisque-score
    idx_name = f"hive-post-data-{post_timestamp.month - 1}-{post_timestamp.year}"
    return [
        { "update" : { "_index" : idx_name, "_id" : post_id } },
        { "doc" : { "avg_clip_vector" : avg_embedding.tolist(), "avg_brisque_score" : avg_brisque_score } }
    ]

def mark_posts_as_proceeded(ids : list, indexes : list) -> None:
    bulk = []
    for idx, id in zip(indexes, ids):
        bulk.append({"update" : {"_index" : idx, "_id" : id}})
        bulk.append({"doc" : {"jobs" : {"img_described" : True}}})

    os_client.bulk(body=bulk)

def send_hearbeat(elapsed_time_ms : int) -> None:
    params = {'msg': 'OK', 'ping' : elapsed_time_ms}

    if CLIP_WORKER_HEARBEAT_URL is not None:
        try:
            requests.get(CLIP_WORKER_HEARBEAT_URL, params=params)
        except Exception as e:
            print(f"CANNOT SEND HEARTBEAT: {e}")

def run() -> None:
    while True:
        start_time = time.time()

        # Get batch
        batch = get_next_batch()
        if len(batch) == 0:
            time.sleep(10)
            continue          

        # Prepare Batch-Data for Thread Pool
        pool_img_func_args = []
        for doc in batch:
            timestamp = datetime.strptime(doc["_source"]["timestamp"], "%Y-%m-%dT%H:%M:%S")
            for img_url in doc["_source"]["image"]:
                pool_img_func_args.append((img_url, timestamp))

        pool_post_func_args = []
        for doc in batch:
            timestamp = datetime.strptime(doc["_source"]["timestamp"], "%Y-%m-%dT%H:%M:%S")
            pool_post_func_args.append((doc["_source"]["image"], doc["_id"], timestamp))

        # Process batch in parallel ThreadPool (max. 5 threads))
        bulk = []
        with ThreadPool(NUM_IMAGE_WORKERS) as pool:
            # Process each image
            for result in pool.map(lambda args: process_image(*args), pool_img_func_args):
               bulk += result   

            # Send image-data bulk
            if len(bulk) > 0:
                res = os_client.bulk(body=bulk)
                bulk = []

            # Process each post
            for result in pool.map(lambda args: combine_post_imgs(*args), pool_post_func_args):
                bulk += result   

            # Send post-data bulk
            if len(bulk) > 0:
                res = os_client.bulk(body=bulk)
                bulk = []            


        # Mark posts as proceeded
        ids = [doc["_id"] for doc in batch]
        indexes = [doc["_index"] for doc in batch]
        mark_posts_as_proceeded(ids, indexes)

        # Finished
        elapsed_time = time.time() - start_time
        print(f"Processed {len(batch)}/{TOTAL_POSTS_FOUND} documents with a total of {len(pool_img_func_args)} images in {elapsed_time} seconds")
        send_hearbeat(math.ceil(elapsed_time * 1000))

if __name__ == "__main__":
    run()

# docker run --env-file V:\Projekte\HiveDiscover\Python\docker_variables.env registry.hive-discover.tech/clip-api:0.8