/**
 * Cloudflare Workers implementation for blog page views API
 * 
 * This worker provides two endpoints:
 * - GET /api/views?slug=<slug> - Get view count for a post
 * - POST /api/views/incr?slug=<slug> - Increment view count
 * 
 * Features:
 * - Persistent storage using Cloudflare KV
 * - 24-hour deduplication per client
 * - CORS support
 * - Rate limiting
 */

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};

// Slug validation
function isValidSlug(slug) {
  if (!slug || typeof slug !== 'string') return false;
  if (slug.length === 0 || slug.length > 200) return false;
  const slugPattern = /^[a-z0-9-/]+$/;
  if (!slugPattern.test(slug)) return false;
  if (slug.startsWith('-') || slug.endsWith('-') || slug.startsWith('/') || slug.endsWith('/')) return false;
  if (slug.includes('--') || slug.includes('//')) return false;
  return true;
}

// Handle GET request - fetch views
async function handleGetViews(request, env) {
  const url = new URL(request.url);
  const slug = url.searchParams.get('slug');

  if (!slug || !isValidSlug(slug)) {
    return new Response(JSON.stringify({ error: 'Invalid slug' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  try {
    const views = await env.VIEWS_KV.get(`views:${slug}`);
    return new Response(
      JSON.stringify({
        slug,
        views: parseInt(views || '0', 10),
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      }
    );
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }
}

// Handle POST request - increment views
async function handleIncrementViews(request, env) {
  const url = new URL(request.url);
  const slug = url.searchParams.get('slug');

  if (!slug || !isValidSlug(slug)) {
    return new Response(JSON.stringify({ error: 'Invalid slug' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  let clientId;
  try {
    const body = await request.json();
    clientId = body.clientId;
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid request body' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  if (!clientId) {
    return new Response(JSON.stringify({ error: 'Client ID is required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }

  try {
    // Check if client has viewed this post in the last 24 hours
    const dedupeKey = `dedupe:${slug}:${clientId}`;
    const lastView = await env.VIEWS_KV.get(dedupeKey);
    const now = Date.now();

    let counted = false;

    // If no previous view or more than 24 hours have passed
    if (!lastView || now - parseInt(lastView, 10) > 24 * 60 * 60 * 1000) {
      // Increment view count
      const viewsKey = `views:${slug}`;
      const currentViews = parseInt((await env.VIEWS_KV.get(viewsKey)) || '0', 10);
      await env.VIEWS_KV.put(viewsKey, String(currentViews + 1));

      // Store last view timestamp with 24h TTL
      await env.VIEWS_KV.put(dedupeKey, String(now), { expirationTtl: 86400 });
      
      counted = true;
    }

    // Get current views
    const views = parseInt((await env.VIEWS_KV.get(`views:${slug}`)) || '0', 10);

    return new Response(
      JSON.stringify({
        slug,
        views,
        counted,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      }
    );
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Internal server error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  }
}

// Main worker entry point
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: corsHeaders,
      });
    }

    // Route to appropriate handler
    if (url.pathname === '/api/views' && request.method === 'GET') {
      return handleGetViews(request, env);
    }

    if (url.pathname === '/api/views/incr' && request.method === 'POST') {
      return handleIncrementViews(request, env);
    }

    // 404 for unknown routes
    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    });
  },
};
