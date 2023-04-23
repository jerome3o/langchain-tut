import textwrap

from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate


def get_synopsis_chain() -> LLMChain:
    llm = OpenAI(temperature=0.7)
    template = textwrap.dedent(
        """\
        You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

        Title: {title}
        Era: {era}
        Playwright: This is a synopsis for the above play:"""
    )
    prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=template
    )
    return LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")


def get_review_chain() -> LLMChain:
    # This is an LLMChain to write a review of a play given a synopsis.
    llm = OpenAI(temperature=0.7)
    template = textwrap.dedent(
        """\
        You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
    )
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    return LLMChain(llm=llm, prompt=prompt_template, output_key="review")


def main():
    syn_chain = get_synopsis_chain()
    rev_chain = get_review_chain()

    overall_chain = SequentialChain(
        chains=[syn_chain, rev_chain],
        input_variables=["era", "title"],
        output_variables=["synopsis", "review"],
        verbose=True,
    )

    result = overall_chain(
        {
            "title": "The SuperTuxKart Massacre",
            "era": "CyberPunk",
        }
    )

    print(result)
    print(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
