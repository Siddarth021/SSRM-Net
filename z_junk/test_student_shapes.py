import sys
import os
import unittest
import torch

# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ECG_SSL_KD.models.student.student_model import StudentModel

class TestStudentShapesAndParams(unittest.TestCase):
    def setUp(self):
        self.model = StudentModel(num_classes=5)
        self.batch_size = 4
        self.lead2 = torch.randn(self.batch_size, 1, 5000)
        self.morphology = torch.randn(self.batch_size, 11, 1250)

    def test_shapes(self):
        outputs = self.model(self.lead2, self.morphology)
        
        embedding = outputs["embedding"]
        projected = outputs["projected"]
        logits = outputs["logits"]
        
        # Verify student embedding shape is (B, 64)
        self.assertEqual(embedding.shape, (self.batch_size, 64))
        
        # Verify projection shape is (B, 256)
        self.assertEqual(projected.shape, (self.batch_size, 256))
        
        # Verify logits shape is (B, 5)
        self.assertEqual(logits.shape, (self.batch_size, 5))

    def test_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nStudent Network parameter breakdown:")
        print(f"  Rhythm encoder params: {sum(p.numel() for p in self.model.rhythm_encoder.parameters()):,}")
        print(f"  Morphology encoder params: {sum(p.numel() for p in self.model.morphology_encoder.parameters()):,}")
        print(f"  Fusion MLP params: {sum(p.numel() for p in self.model.fusion.parameters()):,}")
        print(f"  Projection layer params: {sum(p.numel() for p in self.model.projection.parameters()):,}")
        print(f"  Classifier head params: {sum(p.numel() for p in self.model.classifier.parameters()):,}")
        print(f"  Total student params: {total_params:,}")
        
        # Check parameter count is between 80,000 and 150,000
        self.assertTrue(80000 <= total_params <= 150000, 
                        f"Student parameter count ({total_params}) must be between 80K and 150K.")

if __name__ == "__main__":
    unittest.main()
