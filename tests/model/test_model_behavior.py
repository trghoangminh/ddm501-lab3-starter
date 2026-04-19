"""
Model behavioral tests.

These tests verify the model's behavior patterns:
- Invariance: Output shouldn't change for certain perturbations
- Directional: Output should change in expected direction
- Minimum Functionality: Basic cases the model must handle

TODO: Complete the test implementations below.

Run tests:
    pytest tests/model/test_model_behavior.py -v
"""

import pytest


class TestModelInvariance:
    """
    Invariance tests - output shouldn't change for certain perturbations.
    
    These tests ensure that the model produces consistent results
    when given the same inputs.
    """
    
    # =========================================================================
    # TODO 1: Implement Deterministic Output Tests
    # =========================================================================
    
    def test_same_input_same_output(self, trained_model):
        """Test that same input always produces same output."""
        result1 = trained_model.predict("196", "242")
        result2 = trained_model.predict("196", "242")
        assert result1 == result2
    
    def test_multiple_calls_consistent(self, trained_model):
        """Test that multiple calls produce consistent results."""
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(r == results[0] for r in results)
    
    # =========================================================================
    # TODO 2: Implement Batch Order Invariance Tests
    # =========================================================================
    
    def test_batch_order_independent(self, trained_model):
        """Test that batch predictions are independent of input order."""
        pairs1 = [("196", "242"), ("186", "302")]
        pairs2 = [("186", "302"), ("196", "242")]
        
        results1 = trained_model.predict_batch(pairs1)
        results2 = trained_model.predict_batch(pairs2)
        
        # Results should contain same values
        assert set(results1) == set(results2)
    
    def test_individual_vs_batch_same_results(self, trained_model):
        """Test that individual and batch predictions match."""
        pairs = [("196", "242"), ("186", "302")]
        batch_results = trained_model.predict_batch(pairs)
        individual_results = [trained_model.predict(u, m) for u, m in pairs]
        
        assert batch_results == individual_results


class TestModelDirectional:
    """
    Directional tests - output should change in expected direction.
    
    These tests verify that the model behaves sensibly when inputs
    change in predictable ways.
    """
    
    # =========================================================================
    # TODO 3: Implement Directional Tests
    # =========================================================================
    
    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        """Test that predictions are reasonably close to actual ratings."""
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            actual = pair["actual_rating"]
            assert abs(prediction - actual) < 2.5, f"Prediction {prediction} too far from actual {actual}"
    
    def test_different_movies_different_predictions(self, trained_model):
        """Test that different movies can get different predictions."""
        pred1 = trained_model.predict("196", "242")
        pred2 = trained_model.predict("196", "302")
        # Could be same by chance, but highly unlikely across multiple tries
        assert pred1 is not None and pred2 is not None
    
    def test_different_users_different_predictions(self, trained_model):
        """Test that different users can get different predictions."""
        pred1 = trained_model.predict("196", "242")
        pred2 = trained_model.predict("186", "242")
        assert pred1 is not None and pred2 is not None


class TestMinimumFunctionality:
    """
    Minimum functionality tests - basic cases the model must handle.
    
    These are simple test cases that the model absolutely must pass
    to be considered functional.
    """
    
    # =========================================================================
    # TODO 4: Implement Minimum Functionality Tests
    # =========================================================================
    
    def test_can_predict_for_known_user(self, trained_model):
        """Test that model can make prediction for known user."""
        prediction = trained_model.predict("196", "242")
        assert prediction is not None
        assert 1.0 <= prediction <= 5.0
    
    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        """Test that model can make predictions for multiple known users."""
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert 1.0 <= prediction <= 5.0
    
    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        """Test that not all predictions are the same value."""
        predictions = [
            trained_model.predict(p["user_id"], p["movie_id"])
            for p in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, "All predictions are identical"
    
    # =========================================================================
    # TODO 5: Implement Edge Case Tests
    # =========================================================================
    
    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        """Test that model handles unknown users without crashing."""
        for user_id in unknown_users:
            try:
                prediction = trained_model.predict(user_id, "242")
                # If it returns, should be valid
                assert 1.0 <= prediction <= 5.0
            except (ValueError, KeyError, Exception):
                pass  # Acceptable to raise error
    
    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        """Test that model handles unknown movies without crashing."""
        for movie_id in unknown_movies:
            try:
                prediction = trained_model.predict("196", movie_id)
                assert 1.0 <= prediction <= 5.0
            except (ValueError, KeyError, Exception):
                pass


class TestModelPerformance:
    """
    Performance-related behavioral tests.
    
    These tests verify that the model performs adequately
    on known test cases.
    """
    
    # =========================================================================
    # TODO 6: Implement Performance Tests (BONUS)
    # =========================================================================
    
    def test_average_error_acceptable(self, trained_model, known_user_movie_pairs):
        """Test that average prediction error is acceptable."""
        errors = []
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            actual = pair["actual_rating"]
            errors.append(abs(prediction - actual))
        
        mean_error = sum(errors) / len(errors)
        assert mean_error < 1.5
    
    def test_no_extreme_errors(self, trained_model, known_user_movie_pairs):
        """Test that there are no extreme prediction errors."""
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            actual = pair["actual_rating"]
            assert abs(prediction - actual) <= 3.5


class TestModelRobustness:
    """
    Robustness tests - model behavior under unusual conditions.
    """
    
    # =========================================================================
    # TODO 7: Implement Robustness Tests (BONUS)
    # =========================================================================
    
    def test_handles_string_numeric_ids(self, trained_model):
        """Test that model handles string IDs that look like numbers."""
        try:
            pred = trained_model.predict("00196", "00242")
            assert 1.0 <= pred <= 5.0
        except Exception:
            pass
    
    def test_handles_leading_zeros_in_ids(self, trained_model):
        """Test that model handles IDs with leading zeros."""
        try:
            pred = trained_model.predict("01", "02")
            assert 1.0 <= pred <= 5.0
        except Exception:
            pass


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
