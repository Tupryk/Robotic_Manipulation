import numpy as np
import robotic as ry


def generate_shelf(C: ry.Config, pos: np.ndarray):
    # TODO: More efficient piece building, don't repeat pieces!
    small_opening_dims = [.2, .2, .2]
    inner_wall_width = .005

    openings_small = [4, 6]

    w = small_opening_dims[0]*openings_small[0]
    d = w
    h = small_opening_dims[1]*openings_small[1]

    base_height = .05

    C.addFrame("shelf_base") \
        .setPosition(pos + np.array([0., 0., base_height*.5])) \
        .setShape(ry.ST.ssBox, size=[w, d, base_height, 0.005]) \
        .setColor([.8, .8, .8]) \
        .setContact(1)

    C.addFrame("shelf_middle", "shelf_base") \
        .setRelativePosition([0, 0, base_height*.5+h*.5]) \
        .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., inner_wall_width, h, 0.005]) \
        .setColor([1., 1., 0.])

    for s in range(2):
        p = d*.5-small_opening_dims[2]
        p *= 1. if s == 0 else -1.
        C.addFrame(f"shelf_back_{s}", "shelf_base") \
            .setRelativePosition([p, 0, base_height*.5+h*.5]) \
            .setShape(ry.ST.ssBox, size=[inner_wall_width, w, h, 0.005]) \
            .setColor([1., 1., 0.])
        for j in range(openings_small[1]):
            for i in range(openings_small[0]):
                p = small_opening_dims[2]*.5
                p *= 1. if s == 0 else -1.
                opening_pos = np.array([
                    p,
                    i*small_opening_dims[0] - openings_small[0]*small_opening_dims[0]*.5 + small_opening_dims[0]*.5,
                    j*small_opening_dims[1] - openings_small[1]*small_opening_dims[1]*.5 + small_opening_dims[1]*.5
                    ])
                C.addFrame(f"small_box_left_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos - np.array([0., small_opening_dims[0]*.5, 0.])) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1], 0.005]) \
                    .setColor([1., 1., 0.])
                C.addFrame(f"small_box_right_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos - np.array([0., -small_opening_dims[0]*.5, 0.])) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1], 0.005]) \
                    .setColor([1., 1., 0.])
                C.addFrame(f"small_box_top_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos - np.array([0., 0., -small_opening_dims[1]*.5])) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], inner_wall_width, 0.005]) \
                    .setColor([1., 1., 0.])
                C.addFrame(f"small_box_bottom_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos - np.array([0., 0., small_opening_dims[1]*.5])) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], inner_wall_width, 0.005]) \
                    .setColor([1., 1., 0.])
                
                p = -small_opening_dims[2]*.5
                p *= 1. if s == 0 else -1.
                C.addFrame(f"small_box_blocker_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos - np.array([p, 0, small_opening_dims[1]*.5-.015])) \
                    .setShape(ry.ST.ssBox, size=[inner_wall_width, small_opening_dims[0], .03, 0.005]) \
                    .setColor([1., 1., 0.])
                
                C.addFrame(f"small_box_inside_{s}_{i}_{j}", f"shelf_back_{s}") \
                    .setRelativePosition(opening_pos) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], small_opening_dims[1], 0.005]) \
                    .setColor([0., 0., 0., .2]) \
                    .setContact(0)
            
            p = w*.25
            p *= 1. if s == 0 else -1.
            C.addFrame(f"big_box_bottom_{s}_{j}", "shelf_base") \
                .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1]]) \
                .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.005]) \
                .setColor([1., 1., 0.])
            
            C.addFrame(f"big_box_top_{s}_{j}", "shelf_base") \
                .setRelativePosition([0., p, base_height*.5 + (j+1)*small_opening_dims[1]]) \
                .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.005]) \
                .setColor([1., 1., 0.])
            
            C.addFrame(f"big_box_inside_{s}_{j}", "shelf_base") \
                .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1] + small_opening_dims[1]*.5]) \
                .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, small_opening_dims[1], 0.005]) \
                .setColor([0., 0., 0., .2]) \
                .setContact(0)
            
            p = w*.5
            p *= 1. if s == 0 else -1.
            C.addFrame(f"big_box_blocker_{s}_{j}", "shelf_base") \
                .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1]+.015]) \
                .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., inner_wall_width, .03, 0.005]) \
                .setColor([1., 1., 0.])    

def generate_target_box(C: ry.Config, pos: np.ndarray, box_dims: list[float]=[.15, .2, .15]):

    wall_width = .005

    C.addFrame("target_box_inside") \
        .setPosition(pos) \
        .setShape(ry.ST.ssBox, size=[*box_dims, 0.005]) \
        .setColor([0., 1., 0., .3]) \
        .setContact(0)
    
    C.addFrame("target_box_bottom", "target_box_inside") \
        .setRelativePosition([0., 0., -box_dims[2]*.5]) \
        .setShape(ry.ST.ssBox, size=[*box_dims[:2], wall_width, 0.005]) \
        .setColor([0., 0., .7])
    C.addFrame(f"target_box_left", "target_box_inside") \
        .setRelativePosition([0., box_dims[1]*.5, 0.]) \
        .setShape(ry.ST.ssBox, size=[box_dims[0], wall_width, box_dims[2], 0.005]) \
        .setColor([0., 0., .7])
    C.addFrame(f"target_box_right", "target_box_inside") \
        .setRelativePosition([0., -box_dims[1]*.5, 0.]) \
        .setShape(ry.ST.ssBox, size=[box_dims[0], wall_width, box_dims[2], 0.005]) \
        .setColor([0., 0., .7])
    C.addFrame(f"target_box_front", "target_box_inside") \
        .setRelativePosition([box_dims[0]*.5, 0., 0.]) \
        .setShape(ry.ST.ssBox, size=[wall_width, box_dims[1], box_dims[2], 0.005]) \
        .setColor([0., 0., .7])
    C.addFrame(f"target_box_back", "target_box_inside") \
        .setRelativePosition([-box_dims[0]*.5, 0., 0.]) \
        .setShape(ry.ST.ssBox, size=[wall_width, box_dims[1], box_dims[2], 0.005]) \
        .setColor([0., 0., .7])

if __name__ == "__main__":
    C = ry.Config()
    pos = np.array([0., 0., 0.])
    generate_shelf(C, pos)
    C.view(True)
