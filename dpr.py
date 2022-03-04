doc_dir = "PATH_TO_YOUR_DATA_DIR"
train_filename = "TRAIN_FILENAME"
dev_filename = "DEV_FILENAME"

query_model = "facebook/dpr-question_encoder-single-nq-base"
passage_model = "facebook/dpr-ctx_encoder-single-nq-base"

save_dir = "../saved_models/dpr"

# The  DPR format of the original NQ dataset look like:
[
    {
        "dataset": "nq_dev_psgs_w100",
        "question": "who sings does he love me with reba",
        "answers": [
            "Linda Davis"
        ],
        "positive_ctxs": [
            {
                "title": "Does He Love You",
                "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
                "score": 1000,
                "title_score": 1,
                "passage_id": "11828866"
            },
            ...
        ],
        # negative_ctxs not actually used in Haystack's DPR training so we recommend you set it to an empty list
        # DPR is standardly trained using a method known as in-batch negatives. This means that positive contexts for a given query are treated as negative contexts for the other queries in the batch. Doing so allows for a high degree of computational efficiency, thus allowing the model to be trained on large amounts of data.
        # Due to this reason, negative_ctxs is not actually used
        "negative_ctxs": [
            {
                "title": "Cormac McCarthy",
                "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore \u2013 he quit 16 years ago in El Paso, with one of his young",
                "score": 0,
                "title_score": 0,
                "passage_id": "2145653"
            },
            ...
        ],
        "hard_negative_ctxs": [
            {
                "title": "Why Don't You Love Me (Beyonce\u0301 song)",
                "text": "song. According to the lyrics of \"Why Don't You Love Me\", Knowles impersonates a woman who questions her love interest about the reason for which he does not value her fabulousness, convincing him she's the best thing for him as she sings: \"Why don't you love me... when I make me so damn easy to love?... I got beauty... I got class... I got style and I got ass...\". The singer further tells her love interest that the decision not to choose her is \"entirely foolish\". Originally released as a pre-order bonus track on the deluxe edition of \"I Am...",
                "score": 14.678405,
                "title_score": 0,
                "passage_id": "14525568"
            },
            ...
        ]
    },
    ...
]


from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=256,
    #  follow the original DPR parameters for their max passage length but set max query length to 64 since queries are very rarely longer.
)

retriever.train(
    data_dir=doc_dir,
    train_filename=train_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    n_epochs=1,
    batch_size=16,
    grad_acc_steps=8,
    save_dir=save_dir,
    evaluate_every=3000,
    embed_title=True, # When embed_title=True, the document title is prepended to the input text sequence with a [SEP] token between it and document text.
    num_positives=1, # Note that num_positives needs to be lower or equal to the minimum number of positive_ctxs for queries in your data
    num_hard_negatives=1,
)