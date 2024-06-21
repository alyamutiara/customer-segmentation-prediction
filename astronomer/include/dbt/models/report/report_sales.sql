{{ config(
  materialized='table'
) }}
SELECT
  invoice_id,
  datetime_id,
  fi.product_id,
  Description,
  customer_id,
  quantity,
  total_price
FROM {{ ref('fct_invoices') }} fi
LEFT JOIN {{ ref('dim_product') }} dp
  ON fi.product_id = dp.product_id
WHERE customer_id IS NOT NULL
  AND total_price > 0
  AND quantity > 0
ORDER BY customer_id DESC, invoice_id, product_id, datetime_id