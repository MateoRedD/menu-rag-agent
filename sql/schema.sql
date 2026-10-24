-- Run this once in the Supabase SQL Editor.

-- 1. Enable pgvector
create extension if not exists vector;

-- 2. Table that stores each chunk and its embedding.
--    "strategy" lets us store two chunking strategies side by side and compare them.
create table if not exists documents (
  id         bigserial primary key,
  strategy   text not null,
  content    text not null,
  metadata   jsonb default '{}'::jsonb,
  embedding  vector(1536)  -- text-embedding-3-small returns 1536 dimensions
);

-- 3. Row Level Security on, with no policies: only the service_role key (server side) can read/write.
alter table documents enable row level security;

-- 4. Similarity search. "<=>" is cosine distance in pgvector, so 1 - distance = cosine similarity.
create or replace function match_documents(
  query_embedding vector(1536),
  match_count     int,
  filter_strategy text
)
returns table (id bigint, content text, metadata jsonb, similarity float)
language sql stable
as $$
  select id, content, metadata, 1 - (embedding <=> query_embedding) as similarity
  from documents
  where strategy = filter_strategy
  order by embedding <=> query_embedding
  limit match_count;
$$;
