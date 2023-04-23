from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


def main():
    llm = OpenAI(temperature=0.5)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Whats is a good name for an organised crime syndicate that illegally produces {product}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run("air conditioners")
    print(type(output))
    print(output)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
