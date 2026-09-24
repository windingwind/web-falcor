/** Mirrors RenderPasses/SDFEditor/SelectionWheel: a radial menu drawn with Marker2DSet circle sectors. */

import { ExcludeBorderFlags, type Marker2DSet } from "./Marker2DSet.js";

type Vec2 = [number, number];
type Vec4 = [number, number, number, number];

export interface SelectionWheelDesc {
    /** Sectors per group. */
    sectorGroups: number[];
    position: Vec2;
    minRadius: number;
    maxRadius: number;
    baseColor: Vec4;
    highlightColor: Vec4;
    lineColor: Vec4;
    /** Border thickness in pixels. */
    borderWidth: number;
}

export const kInvalidIndex = 0xffffffff;

export class SelectionWheel {
    private desc!: SelectionWheelDesc;

    constructor(private readonly markers: Marker2DSet) {}

    update(mousePos: Vec2, description: SelectionWheelDesc): void {
        this.desc = description;
        const groupCount = this.desc.sectorGroups.length;
        let minSectorAngle = Infinity;
        for (let g = 0; g < groupCount; g++) minSectorAngle = Math.min(minSectorAngle, this.getAngleOfSectorInGroup(g));
        const groupAngle = this.getGroupAngle();
        const halfGroupSpacingAngle = minSectorAngle * 0.05;
        let excludeGroup = kInvalidIndex;
        const { mouseAngle, dirLength } = this.mouseAngleAndDirLength(mousePos);
        const borderColor: Vec4 = [this.desc.lineColor[0], this.desc.lineColor[1], this.desc.lineColor[2], this.desc.borderWidth];

        // Highlight the sector under the mouse.
        if (dirLength >= this.desc.minRadius && dirLength <= this.desc.maxRadius) {
            const { groupIndex, sectorIndex } = this.groupAndSectorIndexFromAngle(mouseAngle);
            excludeGroup = groupIndex;
            const rotation = this.getRotationOfSector(groupIndex, sectorIndex);
            const sectorAngle = this.getAngleOfSectorInGroup(groupIndex);
            const lastSectorIndex = this.desc.sectorGroups[groupIndex]! - 1;
            if (sectorIndex === 0 || sectorIndex === lastSectorIndex) {
                const groupSpacing = sectorIndex === 0 ? halfGroupSpacingAngle : -halfGroupSpacingAngle;
                const offset = sectorIndex === 0 ? sectorAngle : 0;
                this.addCircleSector(groupAngle * groupIndex + offset, groupAngle - sectorAngle, this.desc.baseColor, borderColor, -groupSpacing, false, sectorIndex === 0 ? ExcludeBorderFlags.Left : ExcludeBorderFlags.Right);
                this.addCircleSector(rotation, sectorAngle, this.desc.highlightColor, borderColor, groupSpacing);
            } else {
                this.addCircleSector(rotation, sectorAngle, this.desc.highlightColor, borderColor);
                this.addCircleSector(groupAngle * groupIndex, sectorAngle * sectorIndex, this.desc.baseColor, borderColor, halfGroupSpacingAngle, false, ExcludeBorderFlags.Right);
                this.addCircleSector(groupAngle * groupIndex + sectorAngle * (sectorIndex + 1), sectorAngle * (lastSectorIndex - sectorIndex), this.desc.baseColor, borderColor, -halfGroupSpacingAngle, false, ExcludeBorderFlags.Left);
            }
        }
        for (let g = 0; g < groupCount; g++) {
            if (g !== excludeGroup) this.addCircleSector(groupAngle * g, groupAngle, this.desc.baseColor, borderColor, halfGroupSpacingAngle, true);
        }
    }

    isMouseOnSector(mousePos: Vec2, groupIndex: number, sectorIndex: number): boolean {
        const { mouseAngle, dirLength } = this.mouseAngleAndDirLength(mousePos);
        const rotation = this.getRotationOfSector(groupIndex, sectorIndex);
        const sectorAngle = this.getAngleOfSectorInGroup(groupIndex);
        return rotation <= mouseAngle && mouseAngle <= rotation + sectorAngle && dirLength >= this.desc.minRadius && dirLength <= this.desc.maxRadius;
    }

    /** Mirrors isMouseOnGroup; the sector index is kInvalidIndex outside the group. */
    isMouseOnGroup(mousePos: Vec2, groupIndex: number): { inGroup: boolean; sectorIndex: number } {
        const { mouseAngle, dirLength } = this.mouseAngleAndDirLength(mousePos);
        const minRotation = this.getRotationOfSector(groupIndex, 0);
        const sectorAngle = this.getAngleOfSectorInGroup(groupIndex);
        const maxRotation = minRotation + sectorAngle * this.desc.sectorGroups[groupIndex]!;
        const inGroup = minRotation <= mouseAngle && mouseAngle <= maxRotation && dirLength >= this.desc.minRadius && dirLength <= this.desc.maxRadius;
        return { inGroup, sectorIndex: inGroup ? Math.floor((mouseAngle - minRotation) / sectorAngle) : kInvalidIndex };
    }

    getCenterPositionOfSector(groupIndex: number, sectorIndex: number): Vec2 {
        const angle = this.getRotationOfSector(groupIndex, sectorIndex) + this.getAngleOfSectorInGroup(groupIndex) * 0.5;
        const r = (this.desc.minRadius + this.desc.maxRadius) * 0.5;
        return [this.desc.position[0] + Math.cos(angle) * r, this.desc.position[1] + Math.sin(angle) * r];
    }

    private getAngleOfSectorInGroup(groupIndex: number): number {
        return this.getGroupAngle() / this.desc.sectorGroups[groupIndex]!;
    }

    private getRotationOfSector(groupIndex: number, sectorIndex: number): number {
        return this.getGroupAngle() * groupIndex + this.getAngleOfSectorInGroup(groupIndex) * sectorIndex;
    }

    private getGroupAngle(): number {
        return (2 * Math.PI) / this.desc.sectorGroups.length;
    }

    private mouseAngleAndDirLength(mousePos: Vec2): { mouseAngle: number; dirLength: number } {
        const dx = mousePos[0] - this.desc.position[0];
        const dy = mousePos[1] - this.desc.position[1];
        let mouseAngle = Math.atan2(dy, dx);
        if (mouseAngle < 0) mouseAngle += Math.PI * 2;
        return { mouseAngle, dirLength: Math.hypot(dx, dy) };
    }

    private groupAndSectorIndexFromAngle(angle: number): { groupIndex: number; sectorIndex: number } {
        const groupIndex = Math.floor(angle / this.getGroupAngle());
        const sectorIndex = Math.floor((angle - this.getRotationOfSector(groupIndex, 0)) / this.getAngleOfSectorInGroup(groupIndex));
        return { groupIndex, sectorIndex };
    }

    private addCircleSector(rotation: number, angle: number, color: Vec4, borderColor: Vec4, margin = 0, marginOnBothSides = false, excludeBorderFlags = ExcludeBorderFlags.None): void {
        const kStartOffset = Math.PI / 2;
        const r = kStartOffset - rotation - angle * 0.5 - (marginOnBothSides ? 0 : margin * 0.5);
        this.markers.addCircleSector(this.desc.position, r, angle - Math.abs(marginOnBothSides ? 2 * margin : margin), this.desc.minRadius, this.desc.maxRadius, color, borderColor, excludeBorderFlags);
    }
}
