"""
Tests added after mutation testing of the round-three changes.

Three independent agents injected 524 single-line bugs into the code written in round three, each on its own copy,
and ran the suite against every one. 324 were caught by the tests that existed; 200 were not. Of those,
27 change nothing anyone can observe (dead stores, unreachable branches, a list of one read at index 0 or -1).
The rest are what this file pins: each test below failed on at least one surviving mutant and passes on the code, and
the agent that wrote it ran it both ways.

Several tests use literal limits (20, 50, 300, 500, 2000, 6, 80, 3600) on purpose. The older tests compute their
expected values from the same module constants, so changing a constant could never make them fail.
"""


# ─── state.py ────────────────────────────────────────────────────────────


def test_set_aside_counts_a_file_that_vanished_since_it_was_judged_as_already_gone():
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    sig = st.file_signature()
    st.STATE_FILE.unlink()
    assert st.set_aside(sig) is True
    assert st.set_aside() is True
    assert sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*')) == []


def test_set_aside_succeeds_on_the_last_attempt(monkeypatch):
    import os
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    real, seen = (os.replace, {'n': 0})

    def locked_until_the_last_try(src, dst):
        seen['n'] += 1
        if seen['n'] < 4:
            raise PermissionError('in use')
        return real(src, dst)
    monkeypatch.setattr(st.os, 'replace', locked_until_the_last_try)
    monkeypatch.setattr(st, 'SAVE_ATTEMPTS', 4)
    monkeypatch.setattr(st, 'SAVE_RETRY_DELAY', 0)
    assert st.set_aside() is True
    assert [p.read_text(encoding='utf-8') for p in sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*'))] == ['{ broken']


def test_set_aside_tries_exactly_save_attempts_times_and_waits_between_tries(monkeypatch):
    import pytest
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    calls, sleeps = ([], [])

    def locked(src, dst):
        calls.append(src)
        raise PermissionError('in use')
    monkeypatch.setattr(st.os, 'replace', locked)
    monkeypatch.setattr(st.time, 'sleep', sleeps.append)
    monkeypatch.setattr(st, 'SAVE_ATTEMPTS', 4)
    monkeypatch.setattr(st, 'SAVE_RETRY_DELAY', 0.125)
    with pytest.raises(PermissionError):
        st.set_aside()
    assert len(calls) == 4
    assert sleeps == [0.125] * 3


def test_set_aside_gives_up_at_once_on_an_error_that_is_not_a_lock(monkeypatch):
    import pytest
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    calls = []

    def broken(src, dst):
        calls.append(src)
        raise OSError('disk full')
    monkeypatch.setattr(st.os, 'replace', broken)
    monkeypatch.setattr(st, 'SAVE_RETRY_DELAY', 0)
    with pytest.raises(OSError, match='disk full'):
        st.set_aside()
    assert len(calls) == 1


def test_a_backup_name_carries_the_process_id():
    import os
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    st.set_aside()
    b, = sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*'))
    assert f'-{os.getpid()}-' in b.name


def test_pruning_leaves_a_lookalike_that_only_starts_like_a_backup_name_alone():
    import state as st
    lookalike = st.STATE_FILE.parent / 'noetic_state.json.bak-20200101-000000-1-2.keep'
    lookalike.write_text('precious', encoding='utf-8')
    for i in range(25):
        st.STATE_FILE.write_text(f'{{damaged {i}', encoding='utf-8')
        st.load()
    assert lookalike.read_text(encoding='utf-8') == 'precious'
    assert len([p for p in sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*')) if p != lookalike]) == 20


def test_a_directory_that_cannot_be_listed_does_not_break_the_pruning(monkeypatch):
    from pathlib import Path
    import state as st

    def cannot_list(self, pattern):
        raise PermissionError('cannot list')
    monkeypatch.setattr(Path, 'glob', cannot_list)
    st._prune_backups()


def test_the_sweep_knows_a_temp_name_with_an_underscore_in_its_random_part():
    import os
    import time
    import state as st
    old = st.STATE_FILE.parent / 'noetic_state-ab_d1234.json.tmp'
    old.write_text('{}', encoding='utf-8')
    t = time.time() - 7200
    os.utime(old, (t, t))
    st._sweep_stale_temp_files()
    assert not old.exists()


def test_the_sweep_leaves_a_name_that_only_starts_like_a_temp_file_alone():
    import os
    import time
    import state as st
    odd = st.STATE_FILE.parent / 'noetic_state-abcd1234.json.tmp.json.tmp'
    odd.write_text('{}', encoding='utf-8')
    t = time.time() - 7200
    os.utime(odd, (t, t))
    st._sweep_stale_temp_files()
    assert odd.exists()


def test_the_sweep_age_limit_is_one_hour():
    import os
    import time
    import state as st
    young = st.STATE_FILE.parent / 'noetic_state-aaaa1111.json.tmp'
    old = st.STATE_FILE.parent / 'noetic_state-bbbb2222.json.tmp'
    for p, age in ((young, 3500), (old, 3700)):
        p.write_text('{}', encoding='utf-8')
        t = time.time() - age
        os.utime(p, (t, t))
    st._sweep_stale_temp_files()
    assert young.exists() and (not old.exists())


def test_a_directory_that_cannot_be_listed_does_not_break_the_sweep(monkeypatch):
    from pathlib import Path
    import state as st

    def cannot_list(self, pattern):
        raise PermissionError('cannot list')
    monkeypatch.setattr(Path, 'glob', cannot_list)
    st._sweep_stale_temp_files()


def test_is_blank_copes_with_a_state_that_is_missing_parts():
    import state as st
    assert st.is_blank({}) is False
    s = st.fresh_state()
    s['self_model'] = None
    assert st.is_blank(s) is True
    s = st.fresh_state()
    del s['values']
    assert st.is_blank(s) is False


def test_load_looks_again_up_to_three_times_when_the_file_keeps_changing(monkeypatch):
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    real, seen = (st.set_aside, {'n': 0})

    def changed_twice_then_ours(sig=None):
        seen['n'] += 1
        return False if seen['n'] < 3 else real(sig)
    monkeypatch.setattr(st, 'set_aside', changed_twice_then_ours)
    s = st.load()
    assert seen['n'] == 3 and s['self_model']['name'] == 'Noetic'
    assert [p.read_text(encoding='utf-8') for p in sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*'))] == ['{ broken']


def test_a_file_the_sanitizer_cannot_make_sense_of_is_set_aside_not_a_crash(monkeypatch):
    import state as st
    st.save(st.fresh_state())
    before = st.STATE_FILE.read_text(encoding='utf-8')

    def cannot(raw):
        raise ValueError('cannot make sense of it')
    monkeypatch.setattr(st, '_backfill', cannot)
    s = st.load()
    assert s['self_model']['name'] == 'Noetic'
    assert [p.read_text(encoding='utf-8') for p in sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*'))] == [before]


def test_load_does_not_move_aside_a_file_that_was_replaced_after_it_was_judged_unreadable(monkeypatch):
    import json
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    real_read, seen = (st._read_state_file, {'n': 0})

    def read_then_someone_repairs_the_file():
        out = real_read()
        if seen['n'] == 0:
            seen['n'] += 1
            good = st.fresh_state()
            good['values'] = ['written by the other process']
            st.STATE_FILE.write_text(json.dumps(good), encoding='utf-8')
        return out
    monkeypatch.setattr(st, '_read_state_file', read_then_someone_repairs_the_file)
    s = st.load()
    assert s['values'] == ['written by the other process']
    assert sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*')) == []


def test_a_file_that_cannot_be_moved_aside_is_reported_as_not_overwritten_with_the_cause(monkeypatch):
    import pytest
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')

    def locked(src, dst):
        raise PermissionError('in use')
    monkeypatch.setattr(st.os, 'replace', locked)
    monkeypatch.setattr(st, 'SAVE_ATTEMPTS', 2)
    monkeypatch.setattr(st, 'SAVE_RETRY_DELAY', 0)
    with pytest.raises(OSError, match='not overwritten') as info:
        st.load()
    assert isinstance(info.value.__cause__, PermissionError)
    assert st.STATE_FILE.read_text(encoding='utf-8') == '{ broken' and sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*')) == []


def test_a_failing_cleanup_does_not_mask_the_error_that_made_the_save_fail(monkeypatch):
    import pytest
    import state as st

    def disk_full(a, b):
        raise OSError('disk full')

    def antivirus_has_it(path):
        raise PermissionError('held open by a scanner')
    monkeypatch.setattr(st.os, 'replace', disk_full)
    monkeypatch.setattr(st.os, 'unlink', antivirus_has_it)
    with pytest.raises(OSError, match='disk full'):
        st.save(st.fresh_state())


def test_a_temp_file_that_save_itself_left_behind_is_swept_once_it_is_old(monkeypatch):
    import os
    import time
    import pytest
    import state as st
    with monkeypatch.context() as m:
        m.setattr(st.os, 'replace', lambda a, b: (_ for _ in ()).throw(OSError('killed')))
        m.setattr(st.os, 'unlink', lambda p: None)
        with pytest.raises(OSError):
            st.save(st.fresh_state())
    left, = list(st.STATE_FILE.parent.glob('*.tmp'))
    long_ago = time.time() - 7200
    os.utime(left, (long_ago, long_ago))
    st._sweep_stale_temp_files()
    assert not left.exists()


def test_the_temp_file_is_created_next_to_the_state_file(monkeypatch):
    import os
    from pathlib import Path
    import state as st
    seen, real = ([], os.replace)

    def spy(src, dst):
        seen.append(Path(src).parent)
        return real(src, dst)
    monkeypatch.setattr(st.os, 'replace', spy)
    st.save(st.fresh_state())
    assert seen == [st.STATE_FILE.parent]


def test_save_waits_between_attempts_and_raises_the_last_refusal(monkeypatch):
    import pytest
    import state as st
    calls, sleeps = ([], [])

    def locked(src, dst):
        calls.append(src)
        raise PermissionError('in use')
    monkeypatch.setattr(st.os, 'replace', locked)
    monkeypatch.setattr(st.time, 'sleep', sleeps.append)
    monkeypatch.setattr(st, 'SAVE_ATTEMPTS', 4)
    monkeypatch.setattr(st, 'SAVE_RETRY_DELAY', 0.125)
    with pytest.raises(PermissionError):
        st.save(st.fresh_state())
    assert len(calls) == 4 and sleeps == [0.125] * 3
    assert list(st.STATE_FILE.parent.glob('*.tmp')) == []


def test_parse_int_keeps_numbers_up_to_300_characters_and_reads_longer_ones_as_infinite():
    import math
    import state as st
    assert st._parse_int('12') == 12 and type(st._parse_int('12')) is int
    assert st._parse_int('9' * 300) == int('9' * 300)
    assert st._parse_int('-' + '9' * 299) == -int('9' * 299)
    assert st._parse_int('9' * 301) == math.inf
    assert st._parse_int('-' + '9' * 300) == -math.inf


def test_whole_numbers_in_unknown_keys_come_back_as_whole_numbers():
    import json
    import state as st
    raw = st.fresh_state()
    raw['extra'] = 7
    raw['nested'] = {'k': [1, 2]}
    st.STATE_FILE.write_text(json.dumps(raw), encoding='utf-8')
    s = st.load()
    assert type(s['extra']) is int and [type(v) for v in s['nested']['k']] == [int, int]


def test_bounded_takes_500_values_and_no_more():
    import pytest
    import state as st
    assert st._bounded(list(range(499))) == list(range(499))
    with pytest.raises(ValueError):
        st._bounded(list(range(500)))


def test_bounded_takes_six_levels_and_no_more():
    import pytest
    import state as st
    assert st._bounded([[[[[[1]]]]]]) == [[[[[[1]]]]]]
    with pytest.raises(ValueError):
        st._bounded([[[[[[[1]]]]]]])


def test_bounded_refuses_nan_and_infinity():
    import pytest
    import state as st
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            st._bounded([bad])


def test_bounded_replaces_control_characters_in_text():
    import state as st
    assert st._bounded('x' + chr(27) + '[2Jy') == 'x' + chr(65533) + '[2Jy'


def test_bounded_cuts_text_at_2000_characters():
    import state as st
    assert st._bounded('a' * 2500) == 'a' * 2000


def test_bounded_replaces_control_characters_in_key_names():
    import state as st
    assert list(st._bounded({'a' + chr(27) + 'b': 1})) == ['a' + chr(65533) + 'b']


def test_bounded_cuts_key_names_at_80_characters():
    import state as st
    assert list(st._bounded({'k' * 100: 1})) == ['k' * 80]


def test_a_full_size_state_loads_without_a_warning_about_keys_that_are_not_unknown(caplog):
    import state as st
    import logging
    s = st.fresh_state()
    for i in range(100):
        st.log_interaction(s, f'msg {i}', 'r', 'low', 'pass', domain='py', confidence=0.5)
    st.save(s)
    with caplog.at_level(logging.WARNING, logger='noetic'):
        loaded = st.load()
    assert len(loaded['interaction_log']) == 100
    assert [r.getMessage() for r in caplog.records] == []


def test_unknown_keys_are_cleaned_when_the_file_is_read():
    import json
    import state as st
    raw = st.fresh_state()
    raw['k' * 100] = 1
    raw['a' + chr(27) + 'b'] = 2
    raw['self_model']['s' * 100] = 3
    raw['self_model']['c' + chr(7) + 'd'] = 4
    st.STATE_FILE.write_text(json.dumps(raw), encoding='utf-8')
    s = st.load()
    assert s['k' * 80] == 1 and s['a' + chr(65533) + 'b'] == 2
    assert s['self_model']['s' * 80] == 3 and s['self_model']['c' + chr(65533) + 'd'] == 4


def test_an_unknown_key_named_like_a_self_model_key_survives_at_the_top_level_and_the_other_way_round():
    import json
    import state as st
    raw = st.fresh_state()
    raw['version'] = 'top-level version note'
    raw['self_model']['values'] = ['a self-model note']
    st.STATE_FILE.write_text(json.dumps(raw), encoding='utf-8')
    s = st.load()
    assert s['version'] == 'top-level version note'
    assert s['self_model']['values'] == ['a self-model note']
    assert s['values'] == st.DEFAULT_VALUES


def test_a_damaged_self_model_does_not_pull_the_top_level_keys_into_it():
    import json
    import state as st
    raw = {'self_model': 'oops', 'name': "not the system's name", 'values': ['v'], 'extra_key': 1}
    st.STATE_FILE.write_text(json.dumps(raw), encoding='utf-8')
    s = st.load()
    assert set(s['self_model']) == set(st.DEFAULT_STATE['self_model'])
    assert s['self_model']['name'] == 'Noetic' and s['extra_key'] == 1 and (s['values'] == ['v'])


def test_a_repeated_insight_is_moved_to_the_end_with_a_new_timestamp():
    import state as st
    s = st.fresh_state()
    st.add_insight(s, 'A', 'user')
    st.add_insight(s, 'B', 'user')
    s['insights'][0]['timestamp'] = '2000-01-01T00:00:00'
    assert st.add_insight_ex(s, 'A', 'user') is False
    assert [i['insight'] for i in s['insights']] == ['B', 'A']
    assert s['insights'][-1]['timestamp'] > '2000-01-01T00:00:00'


def test_an_insight_source_is_cleaned_and_cut_to_60_characters():
    import state as st
    s = st.fresh_state()
    st.add_insight(s, 'a note', source='a' + chr(27) + 'b' + 'z' * 100)
    assert s['insights'][-1]['source'] == ('a' + chr(65533) + 'b' + 'z' * 100)[:60]


def test_the_tension_line_is_clipped_to_300_characters():
    import state as st
    s = st.fresh_state()
    s['self_model']['last_review_tension'] = 't' * 400
    line = [ln for ln in st.get_context_summary(s).splitlines() if 'values tension noted' in ln][0]
    assert line == '- values tension noted: ' + 't' * 299 + '…'


def test_a_tension_without_health_text_adds_no_empty_bullet():
    import state as st
    s = st.fresh_state()
    s['self_model']['last_review_tension'] = 'flattery is complied with'
    lines = st.get_context_summary(s).splitlines()
    i = lines.index('## Last self-review')
    assert lines[i + 1:] == ['- values tension noted: flattery is complied with']


def test_set_aside_takes_the_io_lock_so_it_cannot_overlap_a_load():
    import threading
    import state as st
    st.STATE_FILE.write_text('{ broken', encoding='utf-8')
    done = threading.Event()
    t = threading.Thread(target=lambda: (st.set_aside(), done.set()))
    with st._IO_LOCK:
        t.start()
        finished_while_the_lock_was_held = done.wait(0.4)
    t.join(10)
    assert done.is_set()
    assert not finished_while_the_lock_was_held


def test_load_takes_the_io_lock_so_two_recoveries_cannot_overlap():
    import threading
    import state as st
    done = threading.Event()
    t = threading.Thread(target=lambda: (st.load(), done.set()))
    with st._IO_LOCK:
        t.start()
        finished_while_the_lock_was_held = done.wait(0.4)
    t.join(10)
    assert done.is_set()
    assert not finished_while_the_lock_was_held


def test_the_first_load_of_a_new_state_carries_a_creation_time():
    import json
    import state as st
    s = st.load()
    assert s['self_model']['created'] is not None
    assert json.loads(st.STATE_FILE.read_text(encoding='utf-8'))['self_model']['created'] == s['self_model']['created']


def test_the_value_list_holds_exactly_20_values_and_the_insight_list_exactly_50():
    import state as st
    s = st.fresh_state()
    while len(s['values']) < 20:
        assert st.add_value(s, f"value number {len(s['values'])}") is True
    assert st.add_value(s, 'one too many') is False and len(s['values']) == 20
    for i in range(51):
        st.add_insight(s, f'insight {i}')
    assert len(s['insights']) == 50 and s['insights'][0]['insight'] == 'insight 1'


def test_pruning_deletes_the_oldest_backups_whatever_order_the_directory_lists_them_in(monkeypatch):
    from pathlib import Path
    import state as st
    names = [f'noetic_state.json.bak-20260101-000000-1-{i:010d}' for i in range(25)]
    for n in names:
        (st.STATE_FILE.parent / n).write_text('x', encoding='utf-8')
    real_glob = Path.glob
    monkeypatch.setattr(Path, 'glob', lambda self, pattern: iter(sorted(real_glob(self, pattern), reverse=True)))
    st._prune_backups()
    assert sorted((p.name for p in sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*')))) == names[5:]


def test_a_temp_file_that_vanishes_or_is_locked_during_the_sweep_is_ignored(monkeypatch):
    from pathlib import Path
    import state as st
    old = st.STATE_FILE.parent / 'noetic_state-abcd1234.json.tmp'
    old.write_text('{}', encoding='utf-8')
    real_stat = Path.stat

    def stat(self, *a, **kw):
        if self.name == old.name:
            raise FileNotFoundError(self.name)
        return real_stat(self, *a, **kw)
    monkeypatch.setattr(Path, 'stat', stat)
    st._sweep_stale_temp_files()


def test_a_backup_that_cannot_be_deleted_does_not_break_the_recovery(monkeypatch):
    from pathlib import Path
    import state as st
    real_unlink = Path.unlink

    def unlink(self, *a, **kw):
        if '.bak-' in self.name:
            raise PermissionError('held open by a scanner')
        return real_unlink(self, *a, **kw)
    monkeypatch.setattr(Path, 'unlink', unlink)
    for i in range(22):
        st.STATE_FILE.write_text(f'{{damaged {i}', encoding='utf-8')
        st.load()
    assert len(sorted(st.STATE_FILE.parent.glob('noetic_state.json.bak*'))) == 22


# ─── llm_json.py ────────────────────────────────────────────────────────────


def test_clean_line_keeps_the_whole_text_of_a_long_value_that_is_not_a_string():
    import llm_json as lj
    value = {'k': 'v' * 50}
    assert lj.clean_line(value) == '{"k": "' + 'v' * 50 + '"}'
    assert lj.clean_line(['w'] * 30) == '["w", ' + ', '.join(['"w"'] * 29) + ']'


def test_a_list_of_more_than_one_answer_is_not_an_answer():
    import llm_json as lj
    assert lj.to_tri(['yes', 'no']) is None and lj.to_tri(['yes', 'yes']) is None
    assert lj.to_bool(['yes', 'yes']) is False and lj.to_bool(['no', 'yes'], True) is True


def test_a_tuple_holding_one_answer_reads_as_that_answer():
    import llm_json as lj
    assert lj.to_tri(('yes',)) is True and lj.to_tri(('false',)) is False
    assert lj.to_bool(('yes',)) is True and lj.to_bool(('no',), True) is False


def test_a_real_boolean_inside_a_one_item_list_is_read_as_itself():
    import llm_json as lj
    assert lj.to_tri([False]) is False and lj.to_tri([True]) is True
    assert lj.to_bool([True]) is True and lj.to_bool([False], True) is False


def test_a_first_word_that_is_long_is_still_read():
    import llm_json as lj
    assert lj.to_tri('significantly') is True and lj.to_tri('Significantly different') is True


def test_an_auxiliary_verb_alone_or_without_a_negation_is_not_an_answer():
    import llm_json as lj
    assert lj.to_tri('does') is None and lj.to_tri('is') is None and (lj.to_tri('can') is None)
    assert lj.to_tri('is present') is None and lj.to_tri('does follow') is None
    assert lj.to_tri('does not follow') is False and lj.to_tri('is not present') is False


def test_a_detached_negation_after_an_auxiliary_is_read_as_no():
    import llm_json as lj
    assert lj.to_tri("does n't follow") is False and lj.to_tri("is n't") is False


def test_to_bool_reads_empty_text_as_false_whatever_the_default():
    import llm_json as lj
    assert lj.to_bool('', True) is False and lj.to_bool('   ', True) is False


def test_to_bool_reads_floats_by_value_and_leaves_non_finite_ones_to_the_default():
    import llm_json as lj
    assert lj.to_bool(0.5) is True and lj.to_bool(-0.5) is True and (lj.to_bool(0.0) is False)
    assert lj.to_bool(float('inf')) is False and lj.to_bool(float('-inf')) is False
    assert lj.to_bool(float('nan')) is False
    assert lj.to_bool(float('inf'), True) is True and lj.to_bool(float('nan'), True) is True


def test_an_item_of_nothing_but_replacement_characters_and_spaces_is_dropped():
    import llm_json as lj
    assert lj.to_str_list([chr(0) + ' ' + chr(7), chr(65533) + ' ' + chr(65533), ' ' + chr(1) + ' ', 'kept']) == ['kept']
    assert lj.to_str_list(chr(0) + ' ' + chr(7)) == []
    assert lj.to_str_list([chr(65533) + 'x']) == [chr(65533) + 'x']


def test_control_characters_that_python_calls_whitespace_are_replaced_in_a_line_not_collapsed():
    import llm_json as lj
    for code in (11, 12, 28, 31, 133):
        assert lj.clean_line('a' + chr(code) + 'b') == 'a' + chr(65533) + 'b', hex(code)


def test_control_characters_that_python_calls_whitespace_are_replaced_in_a_list_item_not_collapsed():
    import llm_json as lj
    for code in (11, 12, 28, 31, 133):
        assert lj.to_str_list(['a' + chr(code) + 'b']) == ['a' + chr(65533) + 'b'], hex(code)


def test_a_slash_joins_words_so_yes_slash_no_is_not_an_answer():
    import llm_json as lj
    assert lj.to_tri('yes/no') is None and lj.to_tri('true/false') is None
    assert lj.to_tri('n/a') is False


def test_contractions_are_one_word():
    import llm_json as lj
    assert lj.to_tri("doesn't follow") is False and lj.to_tri("can't tell") is False and (lj.to_tri("isn't") is False)


def test_did_not_reads_as_no_like_does_not():
    import llm_json as lj
    assert lj.to_tri('did not follow') is False and lj.to_tri('did follow') is None


# ─── orchestrator.py ────────────────────────────────────────────────────────────


def test_a_process_with_no_history_still_saves_after_the_file_it_knew_was_deleted():
    import json
    import state as st
    from orchestrator import NoeticOrchestrator
    a = NoeticOrchestrator()                         # counts a session, holds no history, knows the file's signature
    st.STATE_FILE.unlink()
    assert a.add_value("kept") is True
    assert "kept" in json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["values"]


def test_a_file_replaced_after_it_was_judged_unreadable_is_left_alone(monkeypatch):
    import json
    import state as st
    from orchestrator import NoeticOrchestrator
    a = NoeticOrchestrator()
    a.run("one")
    st.STATE_FILE.write_text("{oops", encoding="utf-8")
    real = st.try_load

    def unreadable_then_replaced():
        loaded = real()                              # None: the file is damaged
        other = st.fresh_state()
        other["values"].append("written by the other process")
        st.save(other)                               # right after our read, someone replaces it with a good file
        return loaded

    monkeypatch.setattr(st, "try_load", unreadable_then_replaced)
    a.snapshot()
    assert list(st.STATE_FILE.parent.glob("noetic_state.json.bak*")) == []     # the good file was not moved aside
    assert "written by the other process" in json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["values"]


def test_a_read_reports_a_damaged_file_that_cannot_be_moved_aside(monkeypatch):
    import pytest
    import os
    from pathlib import Path
    import state as st
    from orchestrator import NoeticOrchestrator
    a = NoeticOrchestrator()
    a.run("one")
    st.STATE_FILE.write_text("{oops", encoding="utf-8")
    real = os.replace

    def locked(src, dst):
        if Path(src) == st.STATE_FILE:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(st.os, "replace", locked)
    monkeypatch.setattr(st, "SAVE_ATTEMPTS", 3)
    monkeypatch.setattr(st, "SAVE_RETRY_DELAY", 0)
    with pytest.raises(OSError):
        a.snapshot()                                 # not swallowed: the file is never written over
    assert st.STATE_FILE.read_text(encoding="utf-8") == "{oops"


def test_a_write_back_that_loses_a_race_is_retried_at_the_next_look_not_raised(monkeypatch):
    import json
    import state as st
    from orchestrator import NoeticOrchestrator
    a = NoeticOrchestrator()
    a.run("one")
    st.STATE_FILE.unlink()
    real, seen = st.save, {"n": 0}

    def race(state, *args, **kw):
        seen["n"] += 1
        if seen["n"] == 1:
            raise st.StaleState("someone saved while we repaired")
        return real(state, *args, **kw)

    monkeypatch.setattr(st, "save", race)
    assert a.snapshot()["self_model"]["total_interactions"] == 1       # no exception
    a.snapshot()                                                      # the next look repairs the file
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self_model"]["total_interactions"] == 1


def test_a_change_that_keeps_meeting_a_newer_file_is_tried_twelve_times(orch, monkeypatch):
    import pytest
    import state as st
    saves = []

    def always_stale(state, *a, **kw):
        saves.append(1)
        raise st.StaleState("changed again")

    monkeypatch.setattr(st, "save", always_stale)
    with pytest.raises(OSError, match="kept changing"):
        orch.add_value("x")
    assert len(saves) == 12


def test_each_retry_pauses_and_the_pause_grows_with_the_attempt(orch, monkeypatch):
    import pytest
    import orchestrator as om
    import state as st
    sleeps = []

    def always_stale(state, *a, **kw):
        raise st.StaleState("changed again")

    monkeypatch.setattr(st, "save", always_stale)
    monkeypatch.setattr(om.time, "sleep", sleeps.append)
    monkeypatch.setattr(om.random, "uniform", lambda lo, hi: hi)       # makes the upper bound of each pause visible
    with pytest.raises(OSError):
        orch.add_value("x")
    assert sleeps == [om.COMMIT_BACKOFF * (i + 1) for i in range(12)]


def test_control_characters_and_spaces_alone_are_not_a_message(orch, fake):
    import pytest
    import orchestrator as om
    with pytest.raises(om.InputError):
        orch.run(chr(7) + " " + chr(0))
    assert fake.calls == []


def test_a_request_is_prompted_with_what_another_process_saved(fake):
    from orchestrator import NoeticOrchestrator
    a, b = NoeticOrchestrator(), NoeticOrchestrator()
    assert a.add_value("Value saved by A")
    assert a.add_insight("Insight saved by A")
    b.run("hello")
    assert "- Value saved by A" in fake.calls_for("V")[0]["system"]
    assert "Insight saved by A" in fake.calls_for("C")[0]["system"]


def test_the_assistant_turn_records_the_domain_of_the_answer(orch):
    orch.run("hi")
    assert orch.conversation.turns[1].domain == "math"


def test_the_history_shows_the_risk_values_status_and_confidence_of_the_last_answer(orch):
    orch.run("hi")
    assert "[confabulation_risk=low, values=pass, confidence=80%]" in orch.conversation.history_digest()


def test_the_verbose_review_summary_is_the_health_sentence_not_just_its_label(orch):
    orch.state["self_model"]["total_interactions"] = 9
    res = orch.run("hi", verbose=True)
    assert res["review_summary"] == "healthy - nothing alarming in a small sample."


def test_verbose_carries_the_divergence_notes(orch, fake):
    from fakes import N
    fake.script("N", N(confabulation_risk="medium", divergence_notes=["d1", "d2"]))
    res = orch.run("hi", verbose=True)
    assert res["divergence_notes"] == ["d1", "d2"]


def test_a_manual_review_looks_at_what_another_process_saved(fake):
    from orchestrator import NoeticOrchestrator
    a, b = NoeticOrchestrator(), NoeticOrchestrator()
    b.run("asked through the other process")
    review = a.run_manual_review()
    assert len(fake.calls_for("S")) == 1                              # a stale snapshot has nothing to review
    assert review["applied"]["skipped"] == ""


def test_a_rejected_value_does_not_rewrite_the_file(orch):
    import state as st
    assert orch.add_value("Be kind")
    before = (st.STATE_FILE.read_bytes(), st.file_signature())
    assert orch.add_value("be kind") is False and orch.add_value("none") is False      # duplicate, placeholder
    assert (st.STATE_FILE.read_bytes(), st.file_signature()) == before


def test_an_empty_insight_does_not_rewrite_the_file(orch):
    import state as st
    assert orch.add_insight("a note")
    before = (st.STATE_FILE.read_bytes(), st.file_signature())
    assert orch.add_insight("  ") is False
    assert (st.STATE_FILE.read_bytes(), st.file_signature()) == before


def test_a_cut_off_and_repaired_reply_gets_only_the_cut_off_note():
    import orchestrator as om
    c = {"response": "ans", "truncated": True, "parse_status": "repaired"}
    s = om.synthesize(c, {"confabulation_risk": "low"}, {"status": "pass"})
    assert s["reasons"] == ["response may be cut off (length limit reached)"]
    assert "repaired" not in s["text"]


def test_the_conversation_keeps_the_answer_not_the_annotated_text(orch, fake):
    from fakes import N
    fake.script("N", N(confabulation_risk="high", explanation_matches_reasoning=False))
    res = orch.run("hi")
    assert "confabulation risk" in res["response"]                    # the user is shown the note...
    assert orch.conversation.turns[1].content == "The answer is 42."  # ...but the history holds the answer alone


# ─── agents.py ────────────────────────────────────────────────────────────


def test_a_recommendation_wrapped_in_markdown_marks_still_asks_for_a_change():
    import agents
    for recommendation in ("**Regenerate**", "`regenerate`", "_regenerate_", "*suggest modification*"):
        assert agents._wants_change(recommendation) is True, recommendation


def test_suggest_modifying_asks_for_a_change():
    import agents
    assert agents._wants_change("Suggest modifying the tone") is True


def test_the_notes_alias_is_read_as_values_notes():
    import agents
    assert agents.normalize_v({"status": "veto", "notes": ["n1"]})["values_notes"] == ["n1"]


def test_every_item_of_a_recommendation_list_counts():
    # Pinned by a mutation run, and changed after the run called the first version of this behaviour debatable: a list
    # of two used to be kept whole as text, so ["pass as-is", "regenerate"] with status pass never raised a flag.
    import agents
    out = agents.normalize_v({"status": "pass", "recommendation": ["pass as-is", "regenerate"]})
    assert out["status"] == "flag" and out["recommendation"] == "pass as-is; regenerate"
    assert "recommended 'regenerate'" in out["values_notes"][0]
    out = agents.normalize_v({"status": "pass", "recommendation": ["Suggest modification (tone)", "x"]})
    assert out["status"] == "flag"
    out = agents.normalize_v({"status": "pass", "recommendation": ["pass as-is", "looks fine"]})
    assert out["status"] == "pass" and out["recommendation"] == "pass as-is; looks fine"
    out = agents.normalize_v({"status": "pass", "recommendation": ["regenerate"]})
    assert out["status"] == "flag" and out["recommendation"] == "regenerate"
    out = agents.normalize_v({"status": "pass", "recommendation": []})
    assert out["status"] == "pass" and out["recommendation"] == "pass as-is"


def test_the_recommendation_is_echoed_and_defaults_to_pass_as_is():
    import agents
    assert agents.normalize_v({"status": "veto", "recommendation": "regenerate"})["recommendation"] == "regenerate"
    assert agents.normalize_v({"status": "pass"})["recommendation"] == "pass as-is"


def test_an_unrecognized_status_keeps_the_models_own_notes_after_the_explanation():
    import agents
    out = agents.normalize_v({"status": "maybe", "values_notes": ["flatters the user"]})
    assert out["status"] == "unchecked" and out["check_failed"] is True
    assert out["values_notes"][0].startswith("The values agent returned an unrecognized status")
    assert out["values_notes"][-1] == "flatters the user" and len(out["values_notes"]) == 2


def test_a_flag_that_reports_hidden_uncertainty_is_not_described_as_raised_from_pass():
    import agents
    out = agents.normalize_v({"status": "flag", "hidden_uncertainty_detected": True, "values_notes": ["n"]})
    assert out["status"] == "flag" and out["status_adjusted"] == "" and out["values_notes"] == ["n"]


def test_two_reasons_for_raising_a_pass_are_joined_with_and():
    import agents
    out = agents.normalize_v({"status": "pass", "hidden_uncertainty_detected": True, "recommendation": "regenerate"})
    assert out["values_notes"][0] == ("Status raised from pass to flag because it reported hidden uncertainty and "
                                      "it recommended 'regenerate'.")


def test_the_raise_note_comes_before_the_models_own_notes():
    import agents
    out = agents.normalize_v({"status": "pass", "hidden_uncertainty_detected": True, "values_notes": ["n1"]})
    assert out["values_notes"][0].startswith("Status raised from pass to flag") and out["values_notes"][1:] == ["n1"]


def test_a_values_reply_without_a_sycophancy_flag_reads_as_none_detected():
    import agents
    assert agents.normalize_v({"status": "pass"})["sycophancy_detected"] is False


def test_a_raised_pass_records_why_in_status_adjusted():
    import agents
    out = agents.normalize_v({"status": "pass", "hidden_uncertainty_detected": True})
    assert out["status_adjusted"] == "status raised from pass to flag because it reported hidden uncertainty"


def test_the_assessment_alias_is_read_as_overall_assessment():
    import agents
    assert agents.normalize_n({"confabulation_risk": "high", "assessment": "A"})["overall_assessment"] == "A"


def test_only_a_low_rating_is_raised_for_a_stated_mismatch():
    import agents
    for risk, expected in (("medium", "medium"), ("high", "high"), ("banana", "unknown")):
        out = agents.normalize_n({"explanation_matches_reasoning": False, "confabulation_risk": risk,
                                  "overall_assessment": "As written."})
        assert out["confabulation_risk"] == expected and out["risk_adjusted"] == "", risk
        assert out["overall_assessment"] == "As written."


def test_the_models_own_assessment_stays_in_front_of_the_raise_note():
    import agents
    out = agents.normalize_n({"explanation_matches_reasoning": False, "confabulation_risk": "low",
                              "overall_assessment": "Step 2 is a leap."})
    assert out["overall_assessment"].startswith("Step 2 is a leap. [risk raised from low to medium")


def test_the_divergences_and_notes_aliases_are_read_as_divergence_notes():
    import agents
    assert agents.normalize_n({"confabulation_risk": "high", "divergences": ["d1"]})["divergence_notes"] == ["d1"]
    assert agents.normalize_n({"confabulation_risk": "high", "notes": ["n1"]})["divergence_notes"] == ["n1"]


def test_a_complete_values_reply_that_hit_the_length_limit_is_not_a_pass(fake):
    import agents
    from fakes import V, result
    c = agents.computation_agent("hi", "ctx")
    fake.script("V", result(V(), stop="max_tokens"))
    v = agents.values_agent(c, "hi", ["be honest"])
    assert v["status"] == "unchecked" and v["truncated"] is True and v["parse_status"] == "ok"


def test_a_cleared_values_reply_still_says_it_was_cut_off(fake):
    import agents
    from fakes import result
    c = agents.computation_agent("hi", "ctx")
    fake.script("V", result('{"status": "pass", "values_notes": ["certain about the dates', stop="max_tokens"))
    v = agents.values_agent(c, "hi", ["v"])
    assert v["status"] == "unchecked" and v.get("truncated") is True


def test_a_cleared_values_reply_keeps_how_it_was_parsed(fake):
    import agents
    from fakes import result
    c = agents.computation_agent("hi", "ctx")
    fake.script("V", result('{"status": "pass", "values_notes": ["certain about the dates', stop="max_tokens"))
    v = agents.values_agent(c, "hi", ["v"])
    assert v["status"] == "unchecked" and v.get("parse_how") == "repaired"


def test_a_complete_narration_reply_that_hit_the_length_limit_is_not_low_risk(fake):
    import agents
    from fakes import N, result
    c = agents.computation_agent("hi", "ctx")
    fake.script("N", result(N(), stop="max_tokens"))
    n = agents.narration_agent(c, "hi")
    assert n["confabulation_risk"] == "unknown" and n["truncated"] is True


def test_the_partial_divergence_notes_of_a_cleared_narration_reply_are_kept(fake):
    import agents
    from fakes import result
    cut_n = ('{"explanation_matches_reasoning": true, "confabulation_risk": "low", '
             '"divergence_notes": ["a step is missing"')
    c = agents.computation_agent("hi", "ctx")
    fake.script("N", result(cut_n, stop="max_tokens"))
    n = agents.narration_agent(c, "hi")
    assert n["confabulation_risk"] == "unknown" and n["divergence_notes"] == ["a step is missing"]


def test_a_values_reply_with_more_than_ten_notes_keeps_ten():
    import agents
    out = agents.normalize_v({"status": "flag", "values_notes": [f"note {i}" for i in range(12)]})
    assert out["values_notes"] == [f"note {i}" for i in range(10)]


def test_a_long_narration_assessment_is_cut_to_600_characters():
    import agents
    out = agents.normalize_n({"confabulation_risk": "high", "overall_assessment": "x" * 700})
    assert out["overall_assessment"] == "x" * 600


# ─── self_review.py ────────────────────────────────────────────────────────────


def test_a_non_finding_after_punctuation_is_not_stored_as_a_values_tension():
    import self_review as sr
    review = sr.normalize_review({"overall_health": "watch", "values_tension_pattern": "- insufficient data"})
    assert review["values_tension_pattern"] == ""


def test_not_enough_data_is_not_stored_as_a_values_tension():
    import self_review as sr
    review = sr.normalize_review({"overall_health": "watch", "values_tension_pattern": "Not enough data to say."})
    assert review["values_tension_pattern"] == ""


def test_nothing_to_report_is_not_stored_as_a_values_tension():
    import self_review as sr
    review = sr.normalize_review({"overall_health": "watch", "values_tension_pattern": "Nothing to report this time."})
    assert review["values_tension_pattern"] == ""


def test_the_plural_no_tensions_is_not_stored_as_a_values_tension():
    import self_review as sr
    review = sr.normalize_review({"overall_health": "watch", "values_tension_pattern": "No tensions found."})
    assert review["values_tension_pattern"] == ""


def test_no_obvious_systematic_tension_is_not_stored_as_a_values_tension():
    import self_review as sr
    review = sr.normalize_review({"overall_health": "watch", "values_tension_pattern": "No obvious systematic tension"})
    assert review["values_tension_pattern"] == ""


# ─── api.py ────────────────────────────────────────────────────────────


def test_the_422_body_says_where_and_what_is_wrong(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert len(detail) == 1
    assert detail[0]["type"] == "missing"
    assert detail[0]["loc"] == ["body", "message"]
    assert detail[0]["msg"]


def test_the_422_location_parts_are_all_strings(client):
    r = client.post("/chat", content=b'{"message": ', headers={"content-type": "application/json"})
    assert r.status_code == 422
    loc = r.json()["detail"][0]["loc"]
    assert loc[0] == "body" and all(isinstance(part, str) for part in loc)


def test_the_422_handler_scrubs_control_characters_out_of_messages():
    import asyncio
    import json

    from fastapi.exceptions import RequestValidationError

    import api
    esc = chr(27)
    exc = RequestValidationError([{"type": "value_error", "loc": ("body", "x"), "msg": "bad" + esc + "[2J", "input": "q"}])
    body = json.loads(asyncio.run(api._validation_error(None, exc)).body)
    assert body["detail"][0]["msg"] == "bad" + chr(0xFFFD) + "[2J"


def test_the_too_long_message_counts_the_stored_form(client):
    r = client.post("/insight", json={"insight": "word " * 120})              # 600 raw characters, 599 once spaced
    assert r.status_code == 400 and "599 characters" in r.json()["detail"]


def test_get_self_and_audit_adopt_what_another_process_saved(client):
    from orchestrator import NoeticOrchestrator
    client.post("/chat", json={"message": "hi"})                              # the API now holds an orchestrator
    assert NoeticOrchestrator(count_session=False).add_value("added by the CLI")        # a second process saves
    assert "added by the CLI" in client.get("/self").json()["self_model"]
    assert "added by the CLI" in client.get("/audit/values").json()["report"]


def test_each_audit_name_serves_its_own_report(client):
    titles = {"confabulation": "Confabulation Report", "values": "Values Circuit Report",
              "domains": "Domain Confidence Map", "dark": "Dark Matter Map", "full": "Noetic Field Health Report"}
    client.post("/chat", json={"message": "hi"})
    for name, title in titles.items():
        body = client.get(f"/audit/{name}").json()
        assert body["report_type"] == name, name
        assert body["report"].startswith("=== " + title), name


# ─── cli.py ────────────────────────────────────────────────────────────


def test_the_usage_states_the_real_limits(monkeypatch, capsys):
    import state as st
    from fakes import run_cli
    out = run_cli(monkeypatch, capsys, [], argv=("--help",))
    assert f"add an insight (at most {st.MAX_INSIGHT_CHARS} characters)" in out
    assert f"add a value (at most {st.MAX_VALUE_CHARS} characters)" in out


def test_a_refused_command_line_shows_the_reason_and_the_usage(monkeypatch, capsys):
    import pytest

    from fakes import run_cli
    with pytest.raises(SystemExit) as e:
        run_cli(monkeypatch, capsys, [], argv=["--selff"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "Unknown argument: --selff" in err and "Usage:" in err


def test_cli_accepts_text_of_exactly_the_limit(monkeypatch, capsys):
    from fakes import run_cli
    assert "Insight saved:" in run_cli(monkeypatch, capsys, [], argv=["--insight", "i" * 500])
    assert "Value added:" in run_cli(monkeypatch, capsys, [], argv=["--value", "v" * 300])


def test_several_flags_may_come_before_the_command(monkeypatch, capsys):
    from fakes import run_cli
    assert "Usage:" in run_cli(monkeypatch, capsys, [], argv=("--verbose", "--help"))
    assert "Noetic Self-Model" in run_cli(monkeypatch, capsys, [], argv=("--verbose", "--verbose", "--self"))


def test_self_with_a_help_flag_prints_the_usage_and_reads_nothing(monkeypatch, capsys):
    import state as st
    from fakes import run_cli
    for flag in ("--help", "-h"):
        out = run_cli(monkeypatch, capsys, [], argv=("--self", flag))
        assert "Usage:" in out and "Noetic Self-Model" not in out, flag
        assert not st.STATE_FILE.exists(), flag


def test_a_stdin_without_a_closed_attribute_is_treated_as_open(monkeypatch, capsys):
    import sys

    import cli

    class Bare:
        pass

    monkeypatch.setattr(sys, "argv", ["cli.py"])
    monkeypatch.setattr(sys, "stdin", Bare())
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    cli.main()
    assert "Exiting." in capsys.readouterr().out


def test_a_closed_stdin_is_noticed_before_a_session_is_counted(monkeypatch, capsys):
    import sys

    import pytest

    import cli
    import state as st

    class Shut:
        closed = True

    monkeypatch.setattr(sys, "argv", ["cli.py"])
    monkeypatch.setattr(sys, "stdin", Shut())
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    with pytest.raises(SystemExit) as e:
        cli.main()
    cap = capsys.readouterr()
    assert e.value.code == 1 and "No input is available" in cap.err and cap.out == ""
    assert not st.STATE_FILE.exists()


def test_a_start_up_failure_is_one_line_on_stderr_and_exits_1(monkeypatch, capsys):
    import sys

    import pytest

    import cli
    import state as st

    def deny(*a, **kw):
        raise PermissionError("Access is denied")

    monkeypatch.setattr(st, "save", deny)
    monkeypatch.setattr(sys, "argv", ["cli.py"])
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")
    with pytest.raises(SystemExit) as e:
        cli.main()
    cap = capsys.readouterr()
    assert e.value.code == 1
    assert "Access is denied" in cap.err and "Access is denied" not in cap.out and "Traceback" not in cap.err


def test_a_byte_order_mark_followed_by_spaces_still_hides_nothing(monkeypatch, capsys, fake):
    from fakes import run_cli
    out = run_cli(monkeypatch, capsys, [chr(0xFEFF) + "  exit"])
    assert "Exiting." in out and fake.calls == []


def test_blank_and_padded_lines_are_handled_before_anything_reaches_the_model(monkeypatch, capsys, fake):
    from fakes import run_cli
    out = run_cli(monkeypatch, capsys, ["", "   ", "  --history  ", "hello", "  exit  "])
    assert "cannot be empty" not in out                           # a blank line is skipped, not sent
    assert "Conversation: 0 turns" in out                         # the padded command was recognised
    assert out.count("Noetic: The answer is 42.") == 1 and "Exiting." in out
    assert len(fake.calls_for("C")) == 1


def test_a_failing_prompt_ends_the_session_cleanly(monkeypatch, capsys):
    import sys

    import pytest

    import cli
    for exc in (EOFError, KeyboardInterrupt, RuntimeError, OSError):
        with monkeypatch.context() as m:
            def broken(prompt="", exc=exc):
                raise exc("input is gone")

            m.setattr(sys, "argv", ["cli.py"])
            m.setattr("builtins.input", broken)
            try:
                cli.main()
            except BaseException as e:                           # a KeyboardInterrupt must not escape either
                pytest.fail(f"{type(e).__name__} escaped from main() when input() raised {exc.__name__}")
        assert "Exiting." in capsys.readouterr().out, exc.__name__


def test_a_state_file_that_cannot_be_read_does_not_end_the_session_on_self(monkeypatch, capsys, fake):
    import orchestrator as om
    from fakes import run_cli

    def unreadable(self):
        raise PermissionError("Access is denied")

    monkeypatch.setattr(om.NoeticOrchestrator, "show_self_model", unreadable)
    out = run_cli(monkeypatch, capsys, ["--self", "hello", "exit"])
    assert "[The state file could not be read: Access is denied]" in out
    assert "Noetic: The answer is 42." in out and "Exiting." in out


def test_a_review_that_cannot_be_saved_does_not_end_the_session(monkeypatch, capsys, fake):
    import orchestrator as om
    from fakes import run_cli

    def cannot_save(self):
        raise PermissionError("Access is denied")

    monkeypatch.setattr(om.NoeticOrchestrator, "run_manual_review", cannot_save)
    out = run_cli(monkeypatch, capsys, ["--review", "hello", "exit"])
    assert "[Self-review failed: the result could not be saved: Access is denied]" in out
    assert "Noetic: The answer is 42." in out


def test_the_confidence_note_follows_the_computation_reply_not_the_checks(monkeypatch, capsys, fake):
    from fakes import run_cli
    fake.script("C", '{"reasoning_steps": ["a", "b"], "response": "R", "domain": "math"')      # cut off: no confidence
    out = run_cli(monkeypatch, capsys, ["hello", "exit"], argv=("--verbose",))
    assert "Confidence: not reported" in out and "could not be read as structured output" in out
    assert "no usable number" not in out


# ─── audit.py ────────────────────────────────────────────────────────────


def test_the_health_line_shows_the_reviews_health_text():
    import audit
    import state as st
    s = st.fresh_state()
    s["self_model"].update(last_review_health="watch - thin data", last_review_at=10)
    assert "Last review health:    watch - thin data" in audit.dark_matter_report(s)


def test_the_health_line_says_when_a_review_ran_but_gave_no_health_text():
    import audit
    import state as st
    s = st.fresh_state()
    s["self_model"]["last_review_at"] = 10
    assert "Last review health:    reviewed, but the review gave no health text" in audit.dark_matter_report(s)
