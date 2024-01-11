from scripts.data.doc_q_to_a import *


logger = logging.getLogger(__name__)


DATABRICKS_DOCQ2A_PROMPT = """
The following texts are from Databricks, a company that combines data warehouses and data lakes into a lakehouse architecture.
Your task is to answer user's questions based on the following documents.
----
Source: https://docs.databricks.com/sql/language-manual/index.html
Content:
SHOW VIEWS <sql-ref-syntax-aux-show-views>
### Configuration management
.. toctree::
  :maxdepth: 1
  RESET <sql-ref-syntax-aux-conf-mgmt-reset>
  SET <sql-ref-syntax-aux-conf-mgmt-set>
  SET TIMEZONE <sql-ref-syntax-aux-conf-mgmt-set-timezone>
  USE CATALOG <sql-ref-syntax-ddl-use-catalog>
  USE DATABASE <sql-ref-syntax-ddl-usedb>
  USE SCHEMA <sql-ref-syntax-ddl-use-schema>
### Resource management
**Applies to:** ![check marked yes](/_static/images/icons/check.png) <DBR>
.. toctree::
  :maxdepth: 1
  :glob:
  sql-ref-syntax-aux-resource-*
## Security statements
You use security SQL statements to manage access to data:
.. toctree::
  :maxdepth: 1
  ALTER GROUP <security-alter-group>
  CREATE GROUP <security-create-group>
  DENY <security-deny>
  DROP GROUP <security-drop-group>
  GRANT <security-grant>
  GRANT SHARE <security-grant-share>
  REPAIR PRIVILEGES <security-msck>
----
Answer the following question using the document provided above.
Question: What are some of the security SQL statements available in Databricks?
Answer: Some of the security SQL statements available in Databricks are:
- ALTER GROUP
- CREATE GROUP
- DENY
- DROP GROUP
- GRANT
- GRANT SHARE
- REPAIR PRIVILEGES
----
Source: https://docs.databricks.com/migration-notes/dbr5.5-11.3.html
Content:
* [[SPARK-31061]](https://issues.apache.org/jira/browse/SPARK-31061) [SQL] Provide ability to alter the provider of a table
* [[SPARK-30941]](https://issues.apache.org/jira/browse/SPARK-30941) [PYSPARK] Add a note to asDict to document its behavior when there are duplicate fields
* [[SPARK-30759]](https://issues.apache.org/jira/browse/SPARK-30759) [SC-29843][SQL] Fix cache initialization in StringRegexExpression
* [[SPARK-30998]](https://issues.apache.org/jira/browse/SPARK-30998) [SQL] ClassCastException when a generator having nested inner generators
* [[SPARK-30993]](https://issues.apache.org/jira/browse/SPARK-30993) [SQL] Use its sql type for UDT when checking the type of length (fixed/var) or mutable
* [[SPARK-30813]](https://issues.apache.org/jira/browse/SPARK-30813) [ML] Fix Matrices.sprand comments
* [[SPARK-29419]](https://issues.apache.org/jira/browse/SPARK-29419) [SQL] Fix Encoder thread-safety bug in createDataset(Seq)
----
Answer the following question using the document provided above.
Question: What are some of the changes made in the migration from Databricks runtime 5.5 to 11.3?
Answer: Some of the changes made in the migration from Databricks runtime 5.5 to 11.3 include:
- Providing the ability to alter the provider of a table (SPARK-31061)
- Adding a note to asDict to document its behavior when there are duplicate fields (SPARK-30941)
- Fixing cache initialization in StringRegexExpression (SPARK-30759)
- Addressing a ClassCastException when a generator has nested inner generators (SPARK-30998)
- Using its SQL type for UDT when checking the type of length (fixed/var) or mutable (SPARK-30993)
- Fixing Matrices.sprand comments (SPARK-30813)
- Fixing Encoder thread-safety bug in createDataset(Seq) (SPARK-29419)
----
{fmt_content}
----
Answer the following question using the document provided above.
Question: {question}
Answer:
""".strip()


def main(args: argparse.Namespace):
    random.seed(0)
    data_w_questions = generate_questions_from_dataset(
        args,
        prompt_template = DATABRICKS_DOCQ2A_PROMPT,  # customized
    )
    
    logger.info(f"Generated {len(data_w_questions)} data with answers")
    logger.info(f"Saving to {os.path.join(args.save_dir, args.save_name)}")
    with jsonlines.open(os.path.join(args.save_dir, args.save_name), "w") as fwrite:
        fwrite.write_all(data_w_questions)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Generate (document, question, answer) pairs given a (document, question) pairs. "
        "NOTE: for this script to work properly, we assume data being a list of Dict having keys [gold_docs, questions]" )
    )
    parser = add_parser_arguments(parser)
    args = parse_arguments(parser)

    logger = init_logger(filename=None)

    main(args)