import textwrap

from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

def get_synopsis_chain() -> LLMChain:
    llm = OpenAI(temperature=0.7)
    template = textwrap.dedent("""\
    You are a a playwrite. )given the title of aa play, it is your job to write a synopsis of that title.

    Title: {title}
    Playwrite: This is a synopsis for the above play:""")

    prompt_template = PromptTemplate(input_variables=["title"], template=template)

    return LLMChain(llm=llm, prompt=prompt_template)

def get_review_chain() -> LLMChain:
    llm = OpenAI(temperature=0.7)
    template = textwrap.dedent("""\
    You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:""")
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template)


def main():
    synopsis_chain = get_synopsis_chain()
    review_chain = get_review_chain()
    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain])

    review = overall_chain.run("Tragedy of SuperTuxKart")
    print(review)
    print(review)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
