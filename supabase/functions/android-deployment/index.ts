import 'jsr:@supabase/functions-js/edge-runtime.d.ts';
import { createClient } from 'npm:@supabase/supabase-js@2.49.1';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

// Create a Supabase client with the service role key
const supabase = createClient(
  Deno.env.get('SUPABASE_URL') ?? '',
  Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
);

Deno.serve(async (req) => {
  // Handle CORS preflight request
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: corsHeaders,
    });
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    // Get the authorization header
    const authHeader = req.headers.get('Authorization');
    if (!authHeader) {
      return new Response(JSON.stringify({ error: 'Missing authorization header' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Get the JWT token
    const token = authHeader.replace('Bearer ', '');
    
    // Verify the user
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    
    if (authError || !user) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Parse the request body
    const { 
      action, 
      deploymentId,
      versionName,
      versionCode,
      buildType,
      outputType,
      status,
      track,
      buildLogs,
      errorMessage,
      filePath,
      fileSize
    } = await req.json();

    // Handle different actions
    switch (action) {
      case 'create':
        // Create a new deployment record
        const { data: newDeployment, error: createError } = await supabase
          .from('android_deployments')
          .insert({
            user_id: user.id,
            version_name: versionName,
            version_code: versionCode,
            build_type: buildType,
            output_type: outputType,
            status: status || 'pending',
            track,
            started_at: new Date().toISOString()
          })
          .select()
          .single();

        if (createError) {
          return new Response(JSON.stringify({ error: createError.message }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        }

        return new Response(JSON.stringify({ deployment: newDeployment }), {
          status: 200,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });

      case 'update':
        // Update an existing deployment
        if (!deploymentId) {
          return new Response(JSON.stringify({ error: 'Missing deploymentId' }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        }

        const updateData: Record<string, any> = {};
        
        // Only include fields that are provided
        if (status) updateData.status = status;
        if (buildLogs) updateData.build_logs = buildLogs;
        if (errorMessage) updateData.error_message = errorMessage;
        if (filePath) updateData.file_path = filePath;
        if (fileSize) updateData.file_size = fileSize;
        if (track) updateData.track = track;
        
        // If status is 'completed' or 'failed', set completed_at
        if (status === 'completed' || status === 'failed') {
          updateData.completed_at = new Date().toISOString();
        }

        const { data: updatedDeployment, error: updateError } = await supabase
          .from('android_deployments')
          .update(updateData)
          .eq('id', deploymentId)
          .eq('user_id', user.id)
          .select()
          .single();

        if (updateError) {
          return new Response(JSON.stringify({ error: updateError.message }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        }

        return new Response(JSON.stringify({ deployment: updatedDeployment }), {
          status: 200,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });

      case 'get':
        // Get deployment by ID
        if (!deploymentId) {
          // Get all deployments for user
          const { data: deployments, error: getError } = await supabase
            .from('android_deployments')
            .select('*')
            .eq('user_id', user.id)
            .order('created_at', { ascending: false });

          if (getError) {
            return new Response(JSON.stringify({ error: getError.message }), {
              status: 500,
              headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            });
          }

          return new Response(JSON.stringify({ deployments }), {
            status: 200,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        } else {
          // Get specific deployment
          const { data: deployment, error: getError } = await supabase
            .from('android_deployments')
            .select('*')
            .eq('id', deploymentId)
            .eq('user_id', user.id)
            .single();

          if (getError) {
            return new Response(JSON.stringify({ error: getError.message }), {
              status: 500,
              headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            });
          }

          return new Response(JSON.stringify({ deployment }), {
            status: 200,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          });
        }

      default:
        return new Response(JSON.stringify({ error: 'Invalid action' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});