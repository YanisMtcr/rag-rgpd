PROMPT_STRICT = """Reponds a la question en te basant uniquement sur le contexte suivant.
Si la reponse n'est pas dans le contexte, reponds "Je ne sais pas".

<context>
{context}
</context>

<question>
{question}
</question>
"""


PROMPT_CITATION = """Tu es un assistant DPO. Reponds a la question en t'appuyant uniquement sur le contexte fourni.
Chaque affirmation doit etre suivie d'une citation au format [source: <titre>].
Si le contexte ne suffit pas pour repondre, dis-le explicitement.

<context>
{context}
</context>

<question>
{question}
</question>
"""


PROMPT_STRUCTURED = """Tu es un assistant DPO. Reponds a la question en suivant cette structure:
1. Reponse directe (1 ou 2 phrases)
2. Fondement juridique (article du RGPD ou ligne directrice CNIL)
3. Exemple concret (sanction ou cas cite si disponible)

Utilise uniquement le contexte fourni ci-dessous.

<context>
{context}
</context>

<question>
{question}
</question>
"""


PROMPTS = {
    "strict": PROMPT_STRICT,
    "citation": PROMPT_CITATION,
    "structured": PROMPT_STRUCTURED,
}


def build_context(retrieved_chunks, with_titles=True):
    blocks = []
    for c in retrieved_chunks:
        title = c.metadata.get("title", "")
        if with_titles and title:
            blocks.append(f"[source: {title}]\n{c.text}")
        else:
            blocks.append(c.text)
    return "\n\n".join(blocks)
