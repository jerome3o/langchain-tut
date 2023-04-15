from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def main():
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes open source {product}?",
    )
    print(prompt.format(product="snack bars"))

    llm = OpenAI(temperature=0.5)
    print(llm(prompt.format(product="snack bars")))




if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
