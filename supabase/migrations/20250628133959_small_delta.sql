/*
  # Fix Deployment Progress Tables and Events

  1. New Fields
    - Add visual progress tracking fields to android_deployments
    - Add progress_percentage, current_step, visual_progress, progress_message, last_event_timestamp

  2. New Tables
    - deployment_progress_events for event-based tracking
    - deployment_progress for step-by-step tracking

  3. Security
    - Enable RLS on all tables
    - Add policies for service role and authenticated users
    - Ensure proper access control

  4. Automation
    - Create trigger to update deployment progress
    - Create function to handle progress updates
    - Create view for deployment timeline
*/

-- Add visual progress tracking fields to android_deployments if they don't exist
ALTER TABLE android_deployments 
ADD COLUMN IF NOT EXISTS progress_percentage integer default 0 check (progress_percentage >= 0 and progress_percentage <= 100),
ADD COLUMN IF NOT EXISTS current_step text,
ADD COLUMN IF NOT EXISTS visual_progress integer default 0 check (visual_progress >= 0 and visual_progress <= 100),
ADD COLUMN IF NOT EXISTS progress_message text,
ADD COLUMN IF NOT EXISTS last_event_timestamp timestamp with time zone;

-- Create deployment_progress_events table if it doesn't exist
CREATE TABLE IF NOT EXISTS deployment_progress_events (
  id bigint primary key generated always as identity,
  deployment_id bigint references android_deployments(id) not null,
  event_type text not null check (event_type in ('start', 'progress', 'success', 'error', 'warning', 'info')),
  message text not null,
  percentage integer check (percentage >= 0 and percentage <= 100),
  timestamp timestamp with time zone default now(),
  metadata jsonb
);

-- Enable Row Level Security
ALTER TABLE deployment_progress_events ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist to avoid conflicts
DO $$ 
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress_events' AND policyname = 'Service role can manage progress events'
  ) THEN
    DROP POLICY "Service role can manage progress events" ON deployment_progress_events;
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress_events' AND policyname = 'Users can view their own progress events'
  ) THEN
    DROP POLICY "Users can view their own progress events" ON deployment_progress_events;
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress_events' AND policyname = 'Users can insert their own progress events'
  ) THEN
    DROP POLICY "Users can insert their own progress events" ON deployment_progress_events;
  END IF;
END $$;

-- Create policies for deployment_progress_events
CREATE POLICY "Service role can manage progress events"
  ON deployment_progress_events
  FOR ALL
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Users can view their own progress events"
  ON deployment_progress_events
  FOR SELECT
  TO authenticated
  USING (
    deployment_id IN (
      SELECT id FROM android_deployments 
      WHERE user_id = auth.uid() AND deleted_at IS NULL
    )
  );

CREATE POLICY "Users can insert their own progress events"
  ON deployment_progress_events
  FOR INSERT
  TO authenticated
  WITH CHECK (
    deployment_id IN (
      SELECT id FROM android_deployments 
      WHERE user_id = auth.uid() AND deleted_at IS NULL
    )
  );

-- Create deployment_progress table for step-by-step tracking
CREATE TABLE IF NOT EXISTS deployment_progress (
  id bigint primary key generated always as identity,
  deployment_id bigint references android_deployments(id) not null,
  step text not null,
  progress integer not null default 0 check (progress >= 0 and progress <= 100),
  message text,
  updated_at timestamp with time zone default now()
);

-- Enable Row Level Security
ALTER TABLE deployment_progress ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist to avoid conflicts
DO $$ 
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress' AND policyname = 'Service role can manage deployment progress'
  ) THEN
    DROP POLICY "Service role can manage deployment progress" ON deployment_progress;
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress' AND policyname = 'Users can view their own deployment progress'
  ) THEN
    DROP POLICY "Users can view their own deployment progress" ON deployment_progress;
  END IF;
  
  IF EXISTS (
    SELECT 1 FROM pg_policies 
    WHERE tablename = 'deployment_progress' AND policyname = 'Users can update their own deployment progress'
  ) THEN
    DROP POLICY "Users can update their own deployment progress" ON deployment_progress;
  END IF;
END $$;

-- Create policies for deployment_progress
CREATE POLICY "Service role can manage deployment progress"
  ON deployment_progress
  FOR ALL
  TO public
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Users can view their own deployment progress"
  ON deployment_progress
  FOR SELECT
  TO authenticated
  USING (
    deployment_id IN (
      SELECT id FROM android_deployments 
      WHERE user_id = auth.uid() AND deleted_at IS NULL
    )
  );

CREATE POLICY "Users can update their own deployment progress"
  ON deployment_progress
  FOR UPDATE
  TO authenticated
  USING (
    deployment_id IN (
      SELECT id FROM android_deployments 
      WHERE user_id = auth.uid() AND deleted_at IS NULL
    )
  )
  WITH CHECK (
    deployment_id IN (
      SELECT id FROM android_deployments 
      WHERE user_id = auth.uid() AND deleted_at IS NULL
    )
  );

-- Create function to update deployment progress based on events
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc 
    WHERE proname = 'update_deployment_progress'
  ) THEN
    CREATE OR REPLACE FUNCTION update_deployment_progress()
    RETURNS TRIGGER AS $$
    BEGIN
      -- Update the parent deployment with the latest progress information
      UPDATE android_deployments
      SET 
        visual_progress = NEW.percentage,
        progress_message = NEW.message,
        last_event_timestamp = NEW.timestamp
      WHERE id = NEW.deployment_id;
      
      RETURN NEW;
    END;
    $$ language 'plpgsql';
  ELSE
    -- Update existing function
    CREATE OR REPLACE FUNCTION update_deployment_progress()
    RETURNS TRIGGER AS $$
    BEGIN
      -- Update the parent deployment with the latest progress information
      UPDATE android_deployments
      SET 
        visual_progress = NEW.percentage,
        progress_message = NEW.message,
        last_event_timestamp = NEW.timestamp
      WHERE id = NEW.deployment_id;
      
      RETURN NEW;
    END;
    $$ language 'plpgsql';
  END IF;
END $$;

-- Drop existing trigger if it exists
DO $$ 
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_trigger 
    WHERE tgname = 'update_deployment_progress_on_event'
  ) THEN
    DROP TRIGGER update_deployment_progress_on_event ON deployment_progress_events;
  END IF;
END $$;

-- Create trigger to update deployment progress
CREATE TRIGGER update_deployment_progress_on_event
AFTER INSERT ON deployment_progress_events
FOR EACH ROW
EXECUTE FUNCTION update_deployment_progress();

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_deployment_progress_events_deployment_id 
ON deployment_progress_events(deployment_id);

CREATE INDEX IF NOT EXISTS idx_deployment_progress_events_timestamp 
ON deployment_progress_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_deployment_progress_deployment_id 
ON deployment_progress(deployment_id);

-- Drop existing trigger if it exists
DO $$ 
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_trigger 
    WHERE tgname = 'update_deployment_progress_updated_at'
  ) THEN
    DROP TRIGGER update_deployment_progress_updated_at ON deployment_progress;
  END IF;
END $$;

-- Create trigger to update updated_at column
CREATE TRIGGER update_deployment_progress_updated_at
BEFORE UPDATE ON deployment_progress
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Create or replace view for deployment progress timeline
CREATE OR REPLACE VIEW deployment_progress_timeline AS
SELECT 
  d.id as deployment_id,
  d.version_name,
  d.version_code,
  d.status,
  d.visual_progress,
  d.progress_message,
  e.id as event_id,
  e.event_type,
  e.message,
  e.percentage,
  e.timestamp,
  e.metadata
FROM 
  android_deployments d
LEFT JOIN 
  deployment_progress_events e ON d.id = e.deployment_id
ORDER BY 
  d.id DESC, e.timestamp ASC;