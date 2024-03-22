from langchain_community.document_loaders import AmazonTextractPDFLoader

loader = AmazonTextractPDFLoader("/Users/eishahemchand/Mineral-Intelligence/courthouse docs 2/7019762_181540638_docimage_actual.png")
documents = loader.load()

print(documents)