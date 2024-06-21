-- macros/parse_invoice_date.sql
{% macro parse_invoice_date(invoice_date) %}
  CASE
    WHEN LENGTH({{ invoice_date }}) = 16 THEN
      -- Date format: "MM/DD/YYYY HH:MM"
      PARSE_DATETIME('%m/%d/%Y %H:%M', {{ invoice_date }})
    WHEN LENGTH({{ invoice_date }}) <= 14 THEN
      -- Date format: "MM/DD/YY HH:MM"
      PARSE_DATETIME('%m/%d/%y %H:%M', {{ invoice_date }})
    ELSE
      NULL
  END
{% endmacro %}