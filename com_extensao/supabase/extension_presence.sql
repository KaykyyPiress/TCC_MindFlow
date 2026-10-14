-- Presença da extensão MindFlow no Google Meet.
-- Execute no SQL Editor do Supabase. Pode ser repetido sem duplicar colunas.

alter table public.sessao
  add column if not exists origem text not null default 'site';

alter table public.sessao
  add column if not exists extensao_ativa_em timestamp with time zone;

create index if not exists sessao_extensao_presenca_idx
  on public.sessao (comunicador_id, extensao_ativa_em desc)
  where origem = 'extensao' and status = 'em_andamento';

grant select, insert, update on public.sessao to authenticated;

comment on column public.sessao.origem is
  'Origem da sessão: extensão ou site. Novas reuniões reais são criadas pela extensão.';

comment on column public.sessao.extensao_ativa_em is
  'Horário do último sinal enviado pela extensão ativa no Google Meet.';

select
  exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'sessao' and column_name = 'origem'
  ) as coluna_origem_criada,
  exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'sessao' and column_name = 'extensao_ativa_em'
  ) as coluna_presenca_criada,
  has_table_privilege('authenticated', 'public.sessao', 'update') as extensao_pode_atualizar_presenca;
