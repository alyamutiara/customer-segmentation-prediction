{{ config(
  materialized='table'
) }}
SELECT
  p.product_id,
  p.Description,
  SUM(fi.quantity) AS total_quantity_sold
FROM {{ ref('fct_invoices') }} fi
JOIN {{ ref('dim_product') }} p ON fi.product_id = p.product_id
WHERE fi.product_id IS NOT NULL
  AND quantity > 0
GROUP BY p.product_id, p.Description
ORDER BY total_quantity_sold DESC