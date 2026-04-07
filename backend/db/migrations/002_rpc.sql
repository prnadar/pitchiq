-- PitchIQ: increment_usage RPC function
-- Tracks daily prediction usage per user.
-- Resets count when the date changes (IST timezone).
-- Returns the updated usage count for the day.

-- First, ensure the daily_usage table exists
CREATE TABLE IF NOT EXISTS daily_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    usage_date DATE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Asia/Kolkata')::DATE,
    prediction_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, usage_date)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date
    ON daily_usage(user_id, usage_date);

-- The RPC function called by the backend
CREATE OR REPLACE FUNCTION increment_usage(p_user_id UUID)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_today DATE := (NOW() AT TIME ZONE 'Asia/Kolkata')::DATE;
    v_count INT;
    v_plan TEXT;
    v_limit INT;
BEGIN
    -- Get user's current plan to determine their limit
    SELECT plan INTO v_plan FROM users WHERE id = p_user_id;

    IF v_plan IS NULL THEN
        RETURN json_build_object(
            'success', false,
            'error', 'User not found'
        );
    END IF;

    -- Set daily limit based on plan
    CASE v_plan
        WHEN 'free' THEN v_limit := 2;
        WHEN 'trial' THEN v_limit := 100;
        WHEN 'pro' THEN v_limit := 100;
        WHEN 'expert' THEN v_limit := 1000;
        WHEN 'cancelled' THEN v_limit := 2;
        ELSE v_limit := 2;
    END CASE;

    -- Upsert: insert or increment for today
    INSERT INTO daily_usage (user_id, usage_date, prediction_count, updated_at)
    VALUES (p_user_id, v_today, 1, NOW())
    ON CONFLICT (user_id, usage_date)
    DO UPDATE SET
        prediction_count = daily_usage.prediction_count + 1,
        updated_at = NOW()
    RETURNING prediction_count INTO v_count;

    -- Check if over limit
    IF v_count > v_limit THEN
        RETURN json_build_object(
            'success', false,
            'error', 'Daily limit reached',
            'count', v_count,
            'limit', v_limit,
            'plan', v_plan
        );
    END IF;

    RETURN json_build_object(
        'success', true,
        'count', v_count,
        'limit', v_limit,
        'remaining', v_limit - v_count,
        'plan', v_plan
    );
END;
$$;
