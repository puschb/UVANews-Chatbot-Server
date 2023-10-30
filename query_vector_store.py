from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.vector_stores import FaissVectorStore
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever
import sys,os
import warnings
from tqdm import tqdm
from functools import partialmethod
from transformers.utils import logging
import transformers


  

if __name__ == "__main__":

  warnings.filterwarnings("ignore")
  logging.set_verbosity(transformers.logging.ERROR)
  tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

  query = sys.argv[1]

  system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

  # This will wrap the default prompts that are internal to llama-index
  query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

  #8-bit quantized model (dub)
  query_llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs= {'offload_folder': "offload","load_in_4bit": True},
    # uncomment this if using CUDA to reduce memory usage
    #model_kwargs={"torch_dtype": torch.float16}
)


  embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cuda'})
  )
  service_context = ServiceContext.from_defaults(llm=query_llm,embed_model=embed_model)

  #load indicies from storage
  #build faiss index - using just cpu for now, will change
  vector_store_path = str(os.path.dirname(__file__)) + '/faiss_vector_store'
  faiss_vector_store = FaissVectorStore.from_persist_dir(vector_store_path)
  storage_context = StorageContext.from_defaults(
      vector_store=faiss_vector_store, persist_dir=vector_store_path
  )
  fiass_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)

  #create retriever
  v_retriever = VectorIndexRetriever(
      index=fiass_index,
      similarity_top_k=2,
      vector_store_query_mode="default",
      alpha=None,
      doc_ids=None,
  )

  response_synthesizer = get_response_synthesizer(response_mode='compact', service_context=service_context)
  query_engine = RetrieverQueryEngine(retriever = v_retriever,response_synthesizer=response_synthesizer)

  response = query_engine.query(query)
  print(response)





