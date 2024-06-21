{{ config(
  materialized='table'
) }}

-- dim_product.sql
-- StockCode isn't unique, a product with the same id can have different descriptions and prices
-- Create the dimension table

SELECT
  StockCode AS product_id,
  Description,
  ModeUnitPrice AS UnitPrice
FROM (
  SELECT
    StockCode,
    Description,
    UnitPrice AS ModeUnitPrice,
    ROW_NUMBER() OVER (PARTITION BY StockCode ORDER BY COUNT(*) DESC) AS rn
  FROM
    {{ source('retail', 'raw_invoices') }}
  --WHERE UnitPrice > 0
  GROUP BY
    StockCode, Description, UnitPrice
)
WHERE rn = 1
ORDER BY StockCode
