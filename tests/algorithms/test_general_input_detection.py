"""Generalized auto-inference: dimensions, foreground appearance, archive schema."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from shardelix.workflow import inspect_uploads,run_job
from shardelix.reconstruction.generator import generate


def png(a):
    b=io.BytesIO();Image.fromarray(a).save(b,format="PNG");return b.getvalue()


def texture(h,w):
    rng=np.random.default_rng(92)
    return np.clip(125+700*gaussian_filter(rng.normal(size=(h,w,3)),(1.5,1.5,0)),0,255).astype(np.uint8)


def run(root):
    return run_job(root,{"task":"blind","input_mode":"auto","auto_layout":True},browser=True)


class GeneralInputTests(unittest.TestCase):
    def root(self,t):
        r=Path(t);(r/"upload").mkdir();return r

    def test_rotated_rectangular_tiles_and_uneven_exact_grid(self):
        for shape in ((60,100),(62,101)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as t:
                r=self.root(t);case=r/"generated"
                generate(texture(*shape),case,12,"grid",grid=(3,4),rotations=True)
                zpath=r/"upload/tiles.zip"
                with zipfile.ZipFile(zpath,"w") as z:
                    for p in (case/"pieces").iterdir():z.write(p,"dataset/fragments/"+p.name)
                    # A foreign metadata file is not authoritative geometry.
                    z.writestr("dataset/case.json",json.dumps({"dataset":"other format","number":12}))
                    z.writestr("dataset/groundtruth/original.png",png(texture(20,20)))
                info=inspect_uploads([zpath],True)
                self.assertEqual(info["route"],"grid")
                self.assertIsNotNone(info["metadata_warning"])
                layouts=info["suggested_grid"]["layouts"]
                self.assertIn({"rows":3,"cols":4,"height":shape[0],"width":shape[1]},layouts)
                state=run(r);self.assertEqual(state["diagnostics"]["placed"],12)
                self.assertEqual(state["diagnostics"]["render_collision_pixels"],0)
                self.assertIsNone(state["metrics"])

    def test_malformed_and_stale_metadata_are_reinferred(self):
        for metadata in ("{invalid",json.dumps({"height":60,"width":60,"piece_count":4,
                                                "cut":"grid","rotations":False,"grid":[2,2]})):
            with self.subTest(metadata=metadata),tempfile.TemporaryDirectory() as t:
                r=self.root(t);zpath=r/"upload/tiles.zip";im=texture(40,40)
                with zipfile.ZipFile(zpath,"w") as z:
                    for i in range(4):
                        y,x=divmod(i,2);z.writestr(f"pieces/{i}.png",png(im[y*20:y*20+20,x*20:x*20+20]))
                    z.writestr("case.json",metadata)
                info=inspect_uploads([zpath],True)
                self.assertIsNone(info["metadata"])
                self.assertIn("Ignored incompatible case.json",info["metadata_warning"])
                state=run(r)
                self.assertEqual(state["diagnostics"]["placed"],4)
                self.assertEqual(state["diagnostics"]["import_geometry_mode"],"inferred-grid")

    def test_generic_solid_backgrounds_use_exact_masks(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);yy,xx=np.indices((72,72))
            mask=(xx-36)**2+(yy-36)**2<24**2
            for i,bg in enumerate(([255,255,255],[0,0,0],[0,255,0],[0,0,255],[130,120,100])):
                a=np.empty((72,72,3),np.uint8);a[:]=bg;a[mask]=texture(72,72)[mask]
                # enclosed pixels matching background are still interior.
                a[33:39,33:39]=bg
                (r/"upload"/f"piece{i}.png").write_bytes(png(a))
            info=inspect_uploads(list((r/"upload").iterdir()),True)
            self.assertEqual(info["route"],"irregular")
            state=run(r);self.assertEqual(state["diagnostics"]["scan"]["detected_pieces"],5)
            report=json.loads((r/"result/scan_report.json").read_text())
            self.assertTrue(all(x["method"]=="solid-background" for x in report["pieces"]))
            self.assertGreaterEqual(min(x["mask_pixels"] for x in report["pieces"]),int(mask.sum())-10)

    def test_mask_alias_and_partial_masks(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);zpath=r/"upload/data.zip"
            y,x=np.indices((55,65));mask=(x-30)**2+(y-27)**2<15**2
            a=np.full((55,65,3),255,np.uint8);a[mask]=[100,40,170]
            with zipfile.ZipFile(zpath,"w") as z:
                for i in range(3):z.writestr(f"dataset/fragments/frag{i}.png",png(a))
                z.writestr("dataset/masks/frag0_mask.png",png(mask.astype(np.uint8)))
                z.writestr("dataset/masks/frag1.png",png(mask.astype(np.uint8)*255))
                z.writestr("dataset/reference/source.png",png(texture(50,50)))
            info=inspect_uploads([zpath],True)
            self.assertEqual(info["observed_count"],3)
            self.assertEqual(info["foreground_mask_count"],2)
            self.assertTrue(info["partial_masks"])
            state=run(r);report=json.loads((r/"result/scan_report.json").read_text())
            self.assertEqual(state["diagnostics"]["scan"]["detected_pieces"],3)
            self.assertEqual([p["method"] for p in report["pieces"]],["alpha","alpha","solid-background"])
            self.assertEqual([p["mask_pixels"] for p in report["pieces"]],[int(mask.sum())]*3)

    def test_variable_width_color_strips(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);im=texture(150,61);widths=[9,14,11,13,14]
            x=0
            for i,w in enumerate(widths):
                a=im[:,x:x+w];x+=w
                (r/"upload"/f"strip{i}.png").write_bytes(png(np.rot90(a,(i*3)%4)))
            info=inspect_uploads(list((r/"upload").iterdir()),True)
            self.assertEqual(info["route"],"opaque-strips")
            state=run(r);d=state["diagnostics"]
            self.assertEqual(d["import_geometry_mode"],"inferred-strips")
            self.assertEqual(d["algorithm"],"unknown-width-color-strip-chain")
            self.assertEqual(d["placed"],5)
            self.assertEqual(d["render_collision_pixels"],0)
            with Image.open(r/"result/assembly.png") as img:self.assertEqual(img.size,(61,150))

    def test_blind_panorama_layout_beats_aspect_prior(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);rng=np.random.default_rng(42)
            a=np.clip(128+600*gaussian_filter(rng.normal(size=(40,120,3)),(1.3,1.3,0)),0,255).astype(np.uint8)
            generate(a,r/"gen",12,"grid",grid=(2,6),rotations=True)
            for p in (r/"gen/pieces").iterdir():(r/"upload"/p.name).write_bytes(p.read_bytes())
            i=inspect_uploads(list((r/"upload").iterdir()),True)
            self.assertEqual(i["suggested_grid"]["layouts"][0]["rows"],3)
            d=run(r)["diagnostics"]
            self.assertEqual(d["import_geometry"]["grid"],[2,6])
            self.assertGreaterEqual(len(d["input_detection"]["layout_trials"]),2)

    def test_arbitrary_angle_alpha_strips(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);generate(texture(160,60),r/"gen",6,"strips",grid=(1,6),
                                    angle_mode="continuous",rotations=True,seed=12)
            for p in (r/"gen/pieces").iterdir():(r/"upload"/p.name).write_bytes(p.read_bytes())
            info=inspect_uploads(list((r/"upload").iterdir()),True)
            self.assertEqual(info["route"],"continuous-strips")
            d=run(r)["diagnostics"]
            self.assertEqual(d["input_detection"]["route"],"continuous-strips")
            self.assertEqual(d["angle_mode"],"continuous")
            self.assertEqual(d["placed"],6)
            self.assertEqual((d["import_geometry"]["width"],d["import_geometry"]["height"]),(60,160))
            self.assertEqual(d["render_collision_pixels"],0)

    def test_256_irregular_masks_budget(self):
        from tests.algorithms.test_irregular_auto_import import fixture_archive
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);fixture_archive(r/"upload"/"large.zip",256,size=18)
            d=run(r)["diagnostics"]
            self.assertEqual(d["scan"]["detected_pieces"],256)
            self.assertLessEqual(d["scan"]["searched_pairs"],256*8)
            self.assertEqual(d["scan"]["possible_pairs"],32640)

    def test_single_full_image_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t);(r/"upload/full.png").write_bytes(png(texture(51,70)))
            i=inspect_uploads(list((r/"upload").iterdir()),True)
            self.assertEqual(i["route"],"grid")
            s=run(r);self.assertEqual(s["diagnostics"]["placed"],1)
            self.assertEqual(s["diagnostics"]["import_geometry"]["width"],70)

    def test_ambiguous_opaque_shapes_are_not_destructively_segmented(self):
        with tempfile.TemporaryDirectory() as t:
            r=self.root(t)
            for i,(h,w) in enumerate(((24,42),(35,51),(31,43))):
                (r/"upload"/f"crop{i}.png").write_bytes(png(texture(h,w)))
            s=run(r);self.assertEqual(s["diagnostics"]["scan"]["detected_pieces"],3)
            report=json.loads((r/"result/scan_report.json").read_text())
            self.assertTrue(all(x["method"]=="opaque-raster-preserved" for x in report["pieces"]))
            self.assertEqual([x["mask_pixels"] for x in report["pieces"]],[24*42,35*51,31*43])


if __name__=="__main__":unittest.main()
