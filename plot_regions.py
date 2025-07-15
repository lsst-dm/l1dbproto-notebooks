
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy
import matplotlib.figure
from matplotlib import pyplot
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import lsst.geom
from lsst import sphgeom


def _uv3d_to_vertex(uv3d: sphgeom.UnitVector3d) -> tuple[float, float, float]:
    return uv3d.x(), uv3d.y(), uv3d.z()


def globe(r: float = 1.0) -> list[tuple]:
    PI = numpy.pi
    phi = numpy.linspace(0., 2 * PI, 100)

    lines = []
    for dec in (-PI/3, -PI/6, 0., PI/6, PI/3):
        rr = r * math.cos(dec)
        z = math.sin(dec)
        lines.append((rr * numpy.sin(phi), rr * numpy.cos(phi), numpy.full_like(phi, z), "#88a"))

    ra0_mer = (numpy.full_like(phi, 0), r * numpy.sin(phi), r * numpy.cos(phi), "#8a8")
    lines.append(ra0_mer)
    for ra in (PI/4, PI/2, PI*4/4):
        mer = (ra0_mer[0]*math.cos(ra) - ra0_mer[1]*math.sin(ra), ra0_mer[0]*math.sin(ra) + ra0_mer[1]*math.cos(ra), ra0_mer[2], ra0_mer[3])
        lines.append(mer)

    lines.append(((0., 1.1), (0., 0.), (0., 0.), "#666"))
    return lines


def polygons_to_poly_radec(
    convex_polygons: Iterable[sphgeom.ConvexPolygon],
    alpha: float = .25,
    facecolors: str = "C1",
    edgecolors: str ="black",
) -> PolyCollection:
    poly_vertices = []
    for polygon in convex_polygons:
        vertices = []
        below_pi, above_pi = False, False
        for vtx in polygon.getVertices():
            lon_lat = sphgeom.LonLat(sphgeom.Vector3d(*_uv3d_to_vertex(vtx)))
            ra, dec = lon_lat.getLon().asRadians(), lon_lat.getLat().asRadians()
            if ra <= math.pi: below_pi = True
            if ra > math.pi: above_pi = True
            vertices.append((ra, dec))
        min_ra = min(ra for ra, dec in vertices)
        max_ra = max(ra for ra, dec in vertices)
        if (max_ra - min_ra) > math.pi or not (above_pi and below_pi):
            vertices = [(ra if ra <= math.pi else ra - 2 * math.pi, dec) for ra, dec in vertices]
        poly_vertices.append(vertices)
    poly = PolyCollection(poly_vertices, alpha=alpha, facecolors=facecolors, edgecolors=edgecolors)
    return poly


def polygons_to_poly3d(
    convex_polygons: Iterable[sphgeom.ConvexPolygon],
    alpha: float = .3,
    shade: bool = True,
    facecolors: str = "C1",
    edgecolors: str = "black",
) -> Poly3DCollection:
    poly_vertices = []
    for polygon in convex_polygons:
        vertices = polygon.getVertices()
        vertices = vertices + [vertices[0]]
        vertices = [_uv3d_to_vertex(vtx) for vtx in vertices]
        poly_vertices.append(vertices)
    poly = Poly3DCollection(
        poly_vertices, alpha=alpha, shade=shade, facecolors=facecolors, edgecolors=edgecolors
    )
    return poly


def plot_polygons_3d(
    day_obs: str | int, polygons: Iterable[sphgeom.ConvexPolygon]
) -> matplotlib.figure.Figure:
    visit_polygons = polygons_to_poly3d(polygons)
    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    for line in globe():
        ax.plot(*line[:3], "-", color=line[3])
    ax.add_collection3d(visit_polygons)
    ax.set_aspect('equalxy')
    ax.set_title(f"{day_obs=}")
    ax.set(xlim3d=(-1, 1), xlabel='X')
    ax.set(ylim3d=(-1, 1), ylabel='Y')
    ax.set(zlim3d=(-1, 1), zlabel='Z')
    return fig

def plot_polygons_2d(
    day_obs: str | int, polygons: Iterable[sphgeom.ConvexPolygon]
) -> matplotlib.figure.Figure:
    visit_polygons = polygons_to_poly_radec(polygons)
    fig = pyplot.figure()
    ax = fig.add_subplot(projection="aitoff")
    ax.add_collection(visit_polygons)
    ax.set_title(f"{day_obs=}")
    ax.grid(True)
    return fig


def padded_region(region: sphgeom.Region, margin_arcsec: float = 20.) -> sphgeom.Region:
    margin = sphgeom.Angle.fromDegrees(margin_arcsec/3600.)
    if isinstance(region, sphgeom.ConvexPolygon):
        # This is an ad-hoc, approximate implementation. It should be good
        # enough for catalog loading, but is not a general-purpose solution.
        center = lsst.geom.SpherePoint(region.getCentroid())
        corners = [lsst.geom.SpherePoint(c) for c in region.getVertices()]
        # Approximate the region as a Euclidian square
        # geom.Angle(sphgeom.Angle) converter not pybind-wrapped???
        diagonal_margin = lsst.geom.Angle(margin.asRadians() * math.sqrt(2.0))
        padded = [c.offset(center.bearingTo(c), diagonal_margin) for c in corners]
        return sphgeom.ConvexPolygon.convexHull([c.getVector() for c in padded])
    elif hasattr(region, "dilatedBy"):
        return region.dilatedBy(margin)
    else:
        return region.getBoundingCircle().dilatedBy(margin)
