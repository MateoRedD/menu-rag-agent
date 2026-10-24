-- Phase 2: orders created by the agent and conversations handed off to a human.
-- Run once in the Supabase SQL Editor, after schema.sql.
-- Prices are whole Colombian pesos (COP has no cents), so integers avoid float rounding.
-- The Python code validates everything first; the CHECK constraints are a second line of defense.

create table if not exists orders (
  id            bigserial primary key,
  customer_name text not null,
  items         jsonb not null,  -- [{"name": "Arepa de huevo", "quantity": 2, "unit_price": 7000}]
  delivery_fee  integer not null default 0 check (delivery_fee >= 0),
  total         integer not null check (total > 0),
  order_type    text not null check (order_type in ('dine_in', 'delivery')),
  address       text,
  status        text not null default 'pending' check (status in ('pending', 'confirmed', 'cancelled')),
  created_at    timestamptz not null default now(),
  check (order_type = 'dine_in' or address is not null)  -- deliveries need an address
);

create table if not exists handoffs (
  id         bigserial primary key,
  reason     text not null check (reason in ('complaint', 'allergy', 'group_booking', 'unknown_answer', 'other')),
  summary    text not null,
  created_at timestamptz not null default now()
);

-- Same rule as documents: RLS on, no policies, so only the server-side secret key can read/write.
alter table orders enable row level security;
alter table handoffs enable row level security;
