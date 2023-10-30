from UVANewsApiModule import update
import asyncio, json, os, warnings, transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import MockLLM
from llama_index.vector_stores import FaissVectorStore
from tqdm import tqdm
from functools import partialmethod
from transformers.utils import logging
from datetime import datetime

async def main():
  warnings.filterwarnings("ignore")
  logging.set_verbosity(transformers.logging.ERROR)
  tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

  documents = []
  most_recent_article_date = None
  _, most_recent_article_url = get_data_status()
  api_response = await update(most_recent_article_url)
  articles = api_response['apiResults']

  for article in articles:
    print(article['date'])
    article_date = convert_to_iso8601(article["date"])
    if most_recent_article_date == None or article_date>most_recent_article_date:
      most_recent_article_date = article_date
      most_recent_article_url = article['url']

    content = article['text']
    article.pop('text')
    article.pop('url')

    description = article['description']
    if len(description)>350:
        article['description'] = description[:350]

    doc = Document(text=content,metadata=article)
    documents.append(doc)

  if most_recent_article_date != None:
    persist_data_status(most_recent_article_date,most_recent_article_url)
  else:
    most_recent_article_date, most_recent_article_url = get_data_status()

  node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=128)
  nodes = node_parser.get_nodes_from_documents(documents=documents, show_progress=True)

  llm = MockLLM()
  embed_model = LangchainEmbedding( 
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cuda'})
    )

  service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model, node_parser=node_parser)

  vector_store_path = str(os.path.dirname(__file__)) + '/faiss_vector_store'
  faiss_vector_store = FaissVectorStore.from_persist_dir(vector_store_path)
  storage_context = StorageContext.from_defaults(
      vector_store=faiss_vector_store, persist_dir=vector_store_path
  )
  fiass_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
  fiass_index.insert_nodes(nodes=nodes,show_progress = True)

   
  fiass_index.storage_context.persist("./faiss_vector_store")

  print(most_recent_article_date)

  

def convert_to_iso8601(input_string):
  input_format = "%m/%d/%Y %I:%M:%S %p"
  dt_object = datetime.strptime(input_string, input_format)
  iso8601_string = dt_object.isoformat()
  return iso8601_string


def get_data_status():
  with open('server_document_status.json',"r",encoding='utf-8') as f:
    status = json.loads(f.read())
  return status['mostRecentDate'], status['mostRecentUrl']
   

def persist_data_status(most_recent_article_date, most_recent_article_url):
  with open('server_document_status.json',"w",encoding='utf-8') as f:
    status_json = json.dumps({
      "mostRecentDate": most_recent_article_date,
      "mostRecentUrl": most_recent_article_url
    })
    f.write(status_json)



asyncio.run(main())