
      merge into `iconic-indexer-418610`.`retail`.`dim_customer_scd` as DBT_INTERNAL_DEST
    using `iconic-indexer-418610`.`retail`.`dim_customer_scd__dbt_tmp` as DBT_INTERNAL_SOURCE
    on DBT_INTERNAL_SOURCE.dbt_scd_id = DBT_INTERNAL_DEST.dbt_scd_id

    when matched
     and DBT_INTERNAL_DEST.dbt_valid_to is null
     and DBT_INTERNAL_SOURCE.dbt_change_type in ('update', 'delete')
        then update
        set dbt_valid_to = DBT_INTERNAL_SOURCE.dbt_valid_to

    when not matched
     and DBT_INTERNAL_SOURCE.dbt_change_type = 'insert'
        then insert (`customer_sid`, `customer_id`, `Country`, `first_transaction`, `last_transaction`, `count_order`, `total_purchase`, `dbt_updated_at`, `dbt_valid_from`, `dbt_valid_to`, `dbt_scd_id`)
        values (`customer_sid`, `customer_id`, `Country`, `first_transaction`, `last_transaction`, `count_order`, `total_purchase`, `dbt_updated_at`, `dbt_valid_from`, `dbt_valid_to`, `dbt_scd_id`)


  