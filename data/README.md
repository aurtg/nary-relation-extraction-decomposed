# Format of input data

We followed the format of the dataset of Peng et al. (2017) to represent document graphs in our datasets. We describe the format in the following example. Datasets are saved as JSON files.

```
# Dataset is a list of document graphs represented by dictionaries as follows:
{
  # Identifier (string) of an article from which the document graph is constructed.
  "article": "74830_19.text",

  # Ordered list of entities (candidate entity tuple) which is associated with the document graph.
  "entities": [
    # Each entity is described as a dictionary as follows:
    {
      # Entity ID (string)
      "id": "06myrv",

      # Position indices of mentions of the entity
      "indices": [4, 140],

      # Example of mentions of the entity.
      "mention": "Hopper"
    }, ...
  ],

  # Name of relation which holds among the entities.
  "relationLabel": "N/A",

  # List of sentences of the document graph
  "sentences": [
    # Each sentence is described as a dictionary as follows:
    {
      # Position indice of the ROOT word of parsed dependency of the sentence.
      "root": 8,

      # List of tokens contained in the sentence.
      "nodes": [
        # Each token is described as a dictionary as follows:
        {
          # Position of the token.
          "index": 0,

          # Position of the token in a sentence.
          "indexInsideSentence": 0,

          # Original token, lemmatized token, POS tag, NER tag
          "label": "For",
          "lemma": "for",
          "postag": "IN",
          "nertag": "O",

          # List of links of document graph which starts from this token.
          "arcs": [
            # Each link is described as a dictionary as follows:
            {
              # Name (or type) of the link.
              "label": "depinv:case",

              # Index of a token to which this link is connected.
              "toIndex": 4
            }
          ]
        }, ...
      ]
    }, ...
  ]
}
```

---

Nanyun Peng, Hoifung Poon, Chris Quirk, Kristina Toutanova, and Wen-tau Yih. 2017. Cross-sentence n-ary relation extraction with graph lstms. Transactions of the Association for Computational Linguistics, 5:101–115.
