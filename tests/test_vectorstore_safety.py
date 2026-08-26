import pytest

import src.vectorstore as vectorstore


class _ExistingCollection:
    pass


class _FakeClient:
    def __init__(self):
        self.deleted = False
        self.created = False

    def get_collection(self, name):
        return _ExistingCollection()

    def delete_collection(self, name):
        self.deleted = True

    def create_collection(self, name):
        self.created = True
        return _ExistingCollection()


def test_reset_refuses_to_delete_active_collection(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(vectorstore, "get_client", lambda: client)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        vectorstore.reset_collection()

    assert client.deleted is False
    assert client.created is False
