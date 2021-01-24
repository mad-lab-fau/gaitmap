import re
from tempfile import TemporaryDirectory

import joblib
import pytest
from joblib import Memory

from gaitmap.base import BaseType


class TestCachingMixin:
    """Test that all caching functionality works if enabled."""

    algorithm_class = None
    __test__ = False

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        pass

    def assert_after_action_instance(self, instance):
        """Test some aspects of the resulting instance to ensure that results retrieved from cache are correct."""
        raise NotImplementedError()

    def test_memory_as_params(self, after_action_instance):
        assert hasattr(after_action_instance, "memory")
        assert "memory" in after_action_instance.get_params()

    def test_cached_call_works(self, after_action_instance, capsys):
        parameters = after_action_instance.get_other_params()
        algo = after_action_instance.clone()



        # Call everything without caching
        results = getattr(algo, after_action_instance._action_method)(**parameters)

        # We don't expect any print output from caching:
        assert capsys.readouterr().out.count("[Memory]") == 0
        # Just make sure it actually works before testing
        self.assert_after_action_instance(results)

        # Store some info to compare later:
        algo_id = id(results)
        algo_json = results.to_json()
        algo_hash = joblib.hash(results)

        # Set up caching
        tmp = TemporaryDirectory()
        algo = algo.set_params(memory=Memory(tmp.name, verbose=2))

        # Call the algo the first time.
        # This time we expect the results to be method to be called and stored
        results = getattr(algo, after_action_instance._action_method)(**parameters)
        assert capsys.readouterr().out.count("[Memory] Calling") > 0
        self.assert_after_action_instance(results)

        after_first_id = id(results)
        after_first_json = results.to_json()
        # Need to remove the memory object. Otherwise the hash is changed
        results.memory = None
        after_first_hash = joblib.hash(results)
        # Re-enable caching
        algo = algo.set_params(memory=Memory(tmp.name, verbose=2))

        # Call algo a second time
        # This time all results should be loaded from cache
        results = getattr(algo, after_action_instance._action_method)(**parameters)
        assert len(re.findall(r"[Memory].*: Loading", capsys.readouterr().out)) > 0
        self.assert_after_action_instance(results)

        after_second_id = id(results)
        after_second_json = results.to_json()
        # Need to remove the memory object. Otherwise the hash is changed
        results.memory = None
        after_second_hash = joblib.hash(results)

        # Clean up tmp dir
        tmp.cleanup()

        # We expect that the results are the same every time and also that we used the exact same object and no
        # copy was created at any point.
        assert algo_id == after_first_id == after_second_id
        assert algo_json == after_first_json == after_second_json
        assert algo_hash == after_first_hash == after_second_hash

    def test_cached_json_export(self):
        """Test that there is a warning on json export."""
        instance = self.algorithm_class(memory=Memory(None))
        with pytest.warns(UserWarning) as w:
            instance.to_json()

        assert "joblib.Memory" in str(w[0])



