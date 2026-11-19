from unittest.mock import Mock, patch
import sys
from types import SimpleNamespace

import pytest

from agentD.tools import retrieval


@pytest.fixture
def chembl_client(monkeypatch):
    molecule = Mock()
    molecule.filter.return_value = [{'molecule_structures': {'canonical_smiles': 'CCO'}}]
    client = SimpleNamespace(molecule=molecule)
    monkeypatch.setitem(sys.modules, 'chembl_webresource_client.new_client',
                        SimpleNamespace(new_client=client))
    monkeypatch.setattr(retrieval.time, 'sleep', Mock())
    return client


@pytest.mark.parametrize('name', [
    'finasteride', 'finasteride\n', ' finasteride\r\n',
    '"finasteride"\n', " 'finasteride' \n",
])
def test_drug_lookup_normalizes_agent_argument(name, chembl_client):
    assert retrieval.get_drug_smiles.invoke({'drug_name': name}) == 'CCO'
    chembl_client.molecule.filter.assert_called_once_with(pref_name__iexact='finasteride')


def test_missing_drug_returns_false_value(chembl_client):
    chembl_client.molecule.filter.return_value = []
    assert retrieval.get_drug_smiles.invoke({'drug_name': 'unknown'}) is None
    assert chembl_client.molecule.filter.call_count == 1
    retrieval.time.sleep.assert_not_called()


@pytest.mark.parametrize('name', ['', ' \n', 'finasteride\ndutasteride'])
def test_drug_lookup_rejects_empty_or_multiline_name(name, chembl_client):
    with pytest.raises(ValueError, match='drug_name'):
        retrieval.get_drug_smiles.invoke({'drug_name': name})
    chembl_client.molecule.filter.assert_not_called()


def test_schema_initialization_can_succeed_on_fifth_retry(chembl_client, monkeypatch, caplog):
    loader = Mock(side_effect=[Exception('PRIVATE_HTML_ERROR_PAGE')] * 5 + [chembl_client])
    monkeypatch.setattr(retrieval, '_load_chembl_client', loader)
    assert retrieval.get_drug_smiles.invoke({'drug_name': 'finasteride\n'}) == 'CCO'
    assert loader.call_count == 6
    assert retrieval.time.sleep.call_count == 5
    assert 'attempt=5/6' in caplog.text
    assert 'attempt=6/6 event=completed' in caplog.text
    assert 'PRIVATE_HTML_ERROR_PAGE' not in caplog.text


def test_schema_failure_raises_after_exactly_six_attempts(chembl_client, monkeypatch, caplog):
    loader = Mock(side_effect=Exception('PRIVATE_HTML_ERROR_PAGE'))
    monkeypatch.setattr(retrieval, '_load_chembl_client', loader)
    with pytest.raises(RuntimeError, match='ChEMBL client lookup failed after 6 attempts') as error:
        retrieval.get_drug_smiles.invoke({'drug_name': 'finasteride'})
    assert loader.call_count == 6
    assert retrieval.time.sleep.call_count == 5
    assert 'PRIVATE_HTML_ERROR_PAGE' not in str(error.value) + caplog.text


def test_lazy_query_iteration_failure_is_retried(chembl_client):
    def failing_records():
        raise RuntimeError('query failed while iterating')
        yield
    chembl_client.molecule.filter.side_effect = [failing_records(),
        [{'molecule_structures': {'canonical_smiles': 'CCO'}}]]
    assert retrieval.get_drug_smiles.invoke({'drug_name': 'finasteride'}) == 'CCO'
    assert chembl_client.molecule.filter.call_count == 2
    retrieval.time.sleep.assert_called_once_with(2)


def test_sdk_internal_retries_are_disabled(chembl_client):
    from chembl_webresource_client.settings import Settings
    assert retrieval.get_drug_smiles.invoke({'drug_name': 'finasteride'}) == 'CCO'
    assert Settings.Instance().TOTAL_RETRIES == 0


def test_uniprot_optional_names():
    response = Mock()
    response.json.return_value = {"results": [
        {"primaryAccession": "P10415", "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Bcl-2"}}}},
        {"primaryAccession": "A1", "proteinDescription": {
            "submissionNames": [{"fullName": {"value": "Submitted name"}}]}},
        {"primaryAccession": "A2"},
    ]}
    with patch.object(retrieval.requests, "get", return_value=response):
        assert retrieval.get_uniprot_ids.invoke({"protein_name": "BCL2"}) == [
            ("P10415", "Bcl-2"), ("A1", "Submitted name"), ("A2", "A2")]


@pytest.mark.parametrize("accession", [
    "P10415\n", " P10415\r\n", "P10415", '"P10415"\n',
    ' \t"P10415"\r\n', '" P10415\n"',
])
def test_fasta_strips_surrounding_whitespace(accession):
    response = Mock(status_code=200, text=">sp|P10415|\nMABC\n")
    with patch.object(retrieval.requests, "get", return_value=response) as get:
        assert retrieval.fetch_uniprot_fasta.invoke({"uniprot_id": accession}) == response.text
        get.assert_called_once_with("https://rest.uniprot.org/uniprotkb/P10415.fasta", timeout=(10, 60))


@pytest.mark.parametrize("accession", ["P10415-2", "A0A024RBG1", "A0A024RBG1-2"])
def test_fasta_preserves_supported_accessions(accession):
    with patch.object(retrieval.requests, "get", return_value=Mock(status_code=200, text="FASTA")) as get:
        retrieval.fetch_uniprot_fasta.invoke({"uniprot_id": accession})
        assert get.call_args.args[0].endswith("/" + accession + ".fasta")


@pytest.mark.parametrize("accession", ["", "P10415\nOTHER", "../P10415", "P10415?format=json", "accession:P10415"])
def test_fasta_rejects_invalid_ids_before_request(accession):
    with patch.object(retrieval.requests, "get") as get:
        with pytest.raises(ValueError, match="UniProt"):
            retrieval.fetch_uniprot_fasta.invoke({"uniprot_id": accession})
        get.assert_not_called()

