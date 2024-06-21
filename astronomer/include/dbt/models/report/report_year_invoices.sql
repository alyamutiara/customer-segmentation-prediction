{{ config(
  materialized='table'
) }}
SELECT
  dt.year,
  dt.month,
  dt.hour,
  COUNT(DISTINCT fi.invoice_id) AS num_invoices,
  fi.total_price
FROM {{ ref('fct_invoices') }} fi
JOIN {{ ref('dim_datetime') }} dt ON fi.datetime_id = dt.datetime_id
WHERE fi.total_price > 0
GROUP BY dt.year, dt.month, dt.hour, fi.total_price
ORDER BY dt.year, dt.month, dt.hour, fi.total_price