-- MindFlowIA — acesso necessário para a extensão criar e encerrar reuniões.
-- Pode ser executado mais de uma vez no SQL Editor do Supabase.

alter table public.sessao enable row level security;

drop policy if exists sessao_select_propria on public.sessao;
create policy sessao_select_propria
  on public.sessao for select
  to authenticated
  using (comunicador_id = (select auth.uid()));

drop policy if exists sessao_insert_propria on public.sessao;
create policy sessao_insert_propria
  on public.sessao for insert
  to authenticated
  with check (comunicador_id = (select auth.uid()));

drop policy if exists sessao_update_propria on public.sessao;
create policy sessao_update_propria
  on public.sessao for update
  to authenticated
  using (comunicador_id = (select auth.uid()))
  with check (comunicador_id = (select auth.uid()));

grant select, insert, update on public.sessao to authenticated;

-- O Python usa a chave pública somente através destas RPCs. Os GRANTs abaixo
-- são aplicados apenas quando as funções já existem no projeto.
do $$
begin
  if to_regprocedure('public.entrar_sessao(text,text,jsonb)') is not null then
    execute 'grant execute on function public.entrar_sessao(text,text,jsonb) to anon';
  end if;
  if to_regprocedure('public.registrar_inferencia(uuid,text,timestamp with time zone,jsonb,text,real,real,integer)') is not null then
    execute 'grant execute on function public.registrar_inferencia(uuid,text,timestamp with time zone,jsonb,text,real,real,integer) to anon';
  end if;
  if to_regprocedure('public.sair_sessao(uuid,text)') is not null then
    execute 'grant execute on function public.sair_sessao(uuid,text) to anon';
  end if;
end
$$;

-- Diagnóstico final: a primeira coluna confirma que a extensão autenticada
-- pode criar reuniões; as demais confirmam o histórico anônimo do Python.
select
  has_table_privilege('authenticated', 'public.sessao', 'insert') as extensao_pode_criar_reuniao,
  case when to_regprocedure('public.entrar_sessao(text,text,jsonb)') is null then false
    else has_function_privilege('anon', 'public.entrar_sessao(text,text,jsonb)', 'execute') end as anon_pode_entrar,
  case when to_regprocedure('public.registrar_inferencia(uuid,text,timestamp with time zone,jsonb,text,real,real,integer)') is null then false
    else has_function_privilege('anon', 'public.registrar_inferencia(uuid,text,timestamp with time zone,jsonb,text,real,real,integer)', 'execute') end as anon_pode_registrar,
  case when to_regprocedure('public.sair_sessao(uuid,text)') is null then false
    else has_function_privilege('anon', 'public.sair_sessao(uuid,text)', 'execute') end as anon_pode_sair;
