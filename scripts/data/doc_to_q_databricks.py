from scripts.data.doc_to_q import *


DATABRICKS_DOC2Q_PROMPT = """
The following texts are from Databricks, a company that combines data warehouses and data lakes into a lakehouse architecture.
Your task is to create questions that users might ask if they have not read the documentations.
------
Title: https://docs.databricks.com/clusters/custom-containers.html
Content:
## Use an init script
Databricks Container Services clusters enable customers to include init scripts in the Docker container. In most cases, you should avoid init scripts and instead make customizations through Docker directly (using the Dockerfile). However, certain tasks must be executed when the container starts, instead of when the container is built. Use an init script for these tasks.

For example, suppose you want to run a security daemon inside a custom container. Install and build the daemon in the Docker image through your image building pipeline. Then, add an init script that starts the daemon. In this example, the init script would include a line like `systemctl start my-daemon`.

In the API, you can specify init scripts as part of the cluster spec as follows. For more information, see [_](/dev-tools/api/latest/clusters.md#initscriptinfo).

For Databricks Container Services images, you can also store init scripts in DBFS or cloud storage.

The following steps take place when you launch a Databricks Container Services cluster:

#. VMs are acquired from the cloud provider.
#. The custom Docker image is downloaded from your repo.
#. <Databricks> creates a Docker container from the image.
#. <DBR> code is copied into the Docker container.
#. The init scrips are executed. See [_](/clusters/init-scripts.md#execution-order).
------
Create two questions that a user might ask if they have not read these texts. Only create questions that can be answered using the texts above.
Question 1: Can user configure init script on docker pull image cluster?
Question 2: How do I specify init scripts in a cluster spec in Databricks Container Services?
------
Title: https://docs.databricks.com/migration-notes/dbr7.3-11.3.html
Content:
- [[SPARK-38229]](https://issues.apache.org/jira/browse/SPARK-38229) [SQL] Should't check temp/external/ifNotExists with visitReplaceTable when parser
- [[SPARK-34183]](https://issues.apache.org/jira/browse/SPARK-34183) [SS] DataSource V2: Required distribution and ordering in micro-batch execution
- [[SPARK-37932]](https://issues.apache.org/jira/browse/SPARK-37932) [SQL]Wait to resolve missing attributes before applying DeduplicateRelations
- [[SPARK-37904]](https://issues.apache.org/jira/browse/SPARK-37904) [SQL] Improve RebalancePartitions in rules of Optimizer
- [[SPARK-38236]](https://issues.apache.org/jira/browse/SPARK-38236) [SQL][3.2][3.1] Check if table location is absolute by "new Path(locationUri).isAbsolute" in create/alter table
------
Create two questions that a user might ask if they have not read these texts. Only create questions that can be answered using the texts above.
Question 1: What are some improvements made in databricks runtime 7.3 to 11.3?
Question 2: What are some of the SQL issues fixed in databricks runtime 7.3 to 11.3?
------
Title: https://docs.databricks.com/delta/history.html
Content:
## Retrieve Delta table history
You can retrieve information on the operations, user, timestamp, and so on for each write to a Delta table
by running the `history` command. The operations are returned in reverse chronological order. By default table history is retained for 30 days.
```sql
DESCRIBE HISTORY '/data/events/'          -- get the full history of the table
DESCRIBE HISTORY delta.`/data/events/`
DESCRIBE HISTORY '/data/events/' LIMIT 1  -- get the last operation only
DESCRIBE HISTORY eventsTable
```
For Spark SQL syntax details, see [_](/sql/language-manual/delta-describe-history.md).
See the [_](/delta/index.md#delta-api) for Scala/Java/Python syntax details.
[Data Explorer](/data/index.md) provides a visual view of this detailed table information and history for Delta tables. In addition to the table schema and sample data, you can click the **History** tab to see the table history that displays with `DESCRIBE HISTORY`.
## History schema
The output of the `history` operation has the following columns.
| Column | Type | Description |
| --- | --- | ---|
| version | long | Table version generated by the operation. |
| timestamp | timestamp | When this version was committed. |
| userId | string | ID of the user that ran the operation. |
------
Create two questions that a user might ask if they have not read these texts. Only create questions that can be answered using the texts above.
Question 1: How can I check the delta version of a table?
Question 2: Can I see the timestamps of when changes were made to a delta table?
------
{fmt_content}
------
Create two questions that a user might ask if they have not read these texts. Only create questions that can be answered using the texts above.
Question 1:
""".strip()


def databricks_filter_fn(doc: Document):
    if len(doc.page_content.split()) <= 50:
        return False  # remove documents that are too short
    
    # probability to be included: some domains are not very interesting for a user
    include_weights = {
        "migration-notes/": 0.05,
        "release-notes/runtime": 0.05,
        "sql/language-manual": 0.3,
        "kb.databricks.com/": 0.7,  # we don't need to sum to 1.0
    }
    url = doc.metadata['source']

    weight = 1.0
    for key, _weight in include_weights.items():
        if key in url:
            weight = _weight
            break
    
    if random.random() < weight:
        return True
    return False


def main(args: argparse.Namespace):
    """to customize how (doc, q) pairs would be created, simply copy this function over and modify the "# customizable" parts
    """
    random.seed(0)
    if args.mode in ["init_eval_dset", "all"]:
        documents_dataset = create_positive_n_negative_examples(
            args=args,
            filter_fn=databricks_filter_fn  # customized
        )
        logger.info(f"Created {len(documents_dataset)} <gold document, hard negative documents> pairs.")
    if args.mode in ["create_eval_dset", "all"]:
        eval_dataset, test_dataset = create_heldout_test_dset(
            args,
            doc2q_prompt=DATABRICKS_DOC2Q_PROMPT  # customized
        )
        logger.info(f"Number of eval samples: {len(eval_dataset)}")
        logger.info(f"Number of test samples: {len(test_dataset)}")
    if args.mode in ["create_train_dset", "all"]:
        train_dataset = create_train_dset(
            args,
            doc2q_prompt=DATABRICKS_DOC2Q_PROMPT  # customized
        )
        logger.info(f"Number of train samples: {len(train_dataset)}")
    return


if __name__ == "__main__":
    # main script is from doc_to_q, here we just customize the `filter_fn` and the `doc2q_prompt`
    parser = argparse.ArgumentParser(
        description="Generate (document, question) pairs given a (chunked) document database. This can be used for generating both testing (q, doc) pairs AND training (q, doc) pairs."
    )
    parser = add_parser_arguments(parser)
    args = parse_arguments(parser)
    
    logger = init_logger(filename=None)

    main(args)