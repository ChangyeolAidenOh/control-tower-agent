-- mart.known_future_calendar (Stage 3-A, decision log Stage 3)
-- Known published schedule only: date index, weekday, month, events, SNAP.
-- Covers the full M5 calendar range d_1..d_1969 (d_1942..d_1969 = the
-- competition's forecast horizon). No demand, sales, or price columns.
-- Source: raw M5 calendar (loaded verbatim in Stage 1).
--
-- Identifiers confirmed against sql/01_raw_tables.sql (2026-10-09):
-- raw.m5_calendar(d TEXT PK, wday SMALLINT, month SMALLINT,
-- event_name_1/event_type_1/event_name_2/event_type_2 TEXT NULL,
-- snap_ca/snap_tx/snap_wi SMALLINT).

DROP TABLE IF EXISTS mart.known_future_calendar;

CREATE TABLE mart.known_future_calendar AS
WITH cal AS (
    SELECT
        CAST(SUBSTRING(d FROM 3) AS INTEGER)                 AS period_idx,
        wday::int                                            AS wday,
        month::int                                           AS month,
        -- D1 rule: event_type_1 primary slot, fallback to event_type_2
        COALESCE(
            NULLIF(event_type_1, ''),
            NULLIF(event_type_2, ''),
            'none'
        )                                                    AS event_type,
        CASE WHEN NULLIF(event_name_1, '') IS NOT NULL
               OR NULLIF(event_name_2, '') IS NOT NULL
             THEN 1 ELSE 0 END                               AS is_event,
        snap_ca::int                                         AS snap_ca,
        snap_tx::int                                         AS snap_tx,
        snap_wi::int                                         AS snap_wi
    FROM raw.m5_calendar
),
states AS (
    SELECT DISTINCT state_id FROM mart.sku_daily
)
SELECT
    s.state_id,
    c.period_idx,
    c.wday,
    c.month,
    CASE s.state_id
        WHEN 'CA' THEN c.snap_ca
        WHEN 'TX' THEN c.snap_tx
        WHEN 'WI' THEN c.snap_wi
    END                                                      AS snap,
    c.event_type,
    c.is_event
FROM cal c
CROSS JOIN states s;

ALTER TABLE mart.known_future_calendar
    ADD PRIMARY KEY (state_id, period_idx);

-- Sanity: full contiguous range per state, no demand columns by construction.
DO $$
DECLARE
    n_expected INTEGER;
    n_actual   INTEGER;
BEGIN
    SELECT 3 * MAX(period_idx) INTO n_expected FROM mart.known_future_calendar;
    SELECT COUNT(*) INTO n_actual FROM mart.known_future_calendar;
    IF n_actual <> n_expected THEN
        RAISE EXCEPTION 'known_future_calendar incomplete: % rows, expected %',
            n_actual, n_expected;
    END IF;
END $$;
