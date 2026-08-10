-- installments_features.sql
-- Aggregated installment payment features per SK_ID_CURR
-- Source table: installments_payments
-- ============================================================
-- Note on lateness logic:
--   DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT are both NEGATIVE (days before application).
--   A payment is LATE when it was made AFTER the due date, i.e.,
--   DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT (less negative = more recent = later).

SELECT
    SK_ID_CURR,
    COUNT(*)                                        AS INST_COUNT,

    -- Late payment count and ratio
    COUNT(CASE
        WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1
    END)                                            AS INST_LATE_PAYMENT_COUNT,

    CAST(COUNT(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0)                       AS INST_LATE_PAYMENT_RATIO,

    -- Average days late (only for late payments; positive value = days late)
    AVG(CASE
        WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
        THEN DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
    END)                                            AS INST_AVG_DAYS_LATE,

    -- Maximum days late across all installments
    MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)       AS INST_MAX_DAYS_LATE,

    -- Payment difference: positive = overpaid, negative = underpaid
    AVG(AMT_PAYMENT - AMT_INSTALMENT)               AS INST_AVG_PAYMENT_DIFF,

    -- Maximum shortfall (underpayment)
    MAX(AMT_INSTALMENT - AMT_PAYMENT)               AS INST_MAX_PAYMENT_SHORTFALL,

    -- Ratio of installments where applicant underpaid
    CAST(COUNT(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0)                       AS INST_PAYMENT_SHORTFALL_RATIO,

    AVG(AMT_PAYMENT)                                AS INST_AVG_PAYMENT

FROM installments_payments
GROUP BY SK_ID_CURR
