import textwrap

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.memory import SimpleMemory


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


def get_social_media_post_chain() -> LLMChain:
    llm = OpenAI(temperature=0.7)
    template = textwrap.dedent(
        """\
        You are a social media manager for a theater company.  Given the title of play, the era it is set in, the date, time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

        Here is some context about the time and location of the play:
        Date and Time: {time}
        Location: {location}

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:
        {review}

        Social Media Post:
        """
    )
    prompt_template = PromptTemplate(
        input_variables=["synopsis", "review", "time", "location"], template=template
    )
    return LLMChain(llm=llm, prompt=prompt_template, output_key="social_post_text")


def main():
    syn_chain = get_synopsis_chain()
    rev_chain = get_review_chain()
    soc_chain = get_social_media_post_chain()

    overall_chain = SequentialChain(
        chains=[syn_chain, rev_chain, soc_chain],
        memory=SimpleMemory(
            memories={
                "time": "December 25th, 8pm PST",
                "location": "Theater in the Park",
            }
        ),
        input_variables=["era", "title"],
        # here we return multiple variables???
        output_variables=["social_post_text"],
        verbose=True,
    )

    result = overall_chain(
        {
            "title": "The Hungry Tux",
            "era": "80s",
        }
    )

    from pprint import pprint

    pprint(result)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
