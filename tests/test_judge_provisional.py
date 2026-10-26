from castor.judge_provisional import TEMPLATE, parse


def test_parse_official_output():
    r = parse("Harmful request: yes\nResponse refusal: yes\nHarmful response: no")
    assert r == {"harmful_request": True, "refusal": True, "harmful_response": False,
                 "parse_error": False}


def test_parse_failure_is_none_not_compliance():
    assert parse("garbage")["refusal"] is None and parse("garbage")["parse_error"]
    r = parse("Harmful request: yes\nResponse refusal: maybe\nHarmful response: no")
    assert r["refusal"] is None and r["parse_error"]


def test_na_response_fields():
    r = parse("Harmful request: no\nResponse refusal: N/A\nHarmful response: N/A")
    assert r["refusal"] is None and r["harmful_response"] is None


def test_template_has_slots_and_no_bos_literal():
    assert "{prompt}" in TEMPLATE and "{response}" in TEMPLATE
    assert not TEMPLATE.startswith("<s>") and TEMPLATE.endswith("<|assistant|>\n")
