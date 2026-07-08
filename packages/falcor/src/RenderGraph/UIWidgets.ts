/**
 * Minimal widget vocabulary a RenderPass.renderUI() uses to expose controls,
 * mirroring the subset of Falcor's Gui::Widgets that passes actually call. The
 * host (e.g. the Mogwai viewer) provides a concrete DOM/imgui implementation;
 * callbacks fire when the user changes a control so the pass can mutate its state.
 */
export interface UIWidgets {
    /** Static label / read-only status line. */
    text(label: string): void;
    /** Momentary button. */
    button(label: string, onClick: () => void): void;
    /** Boolean toggle. */
    checkbox(label: string, value: boolean, onChange: (value: boolean) => void): void;
    /** Numeric slider over [min, max] with the given step. */
    slider(label: string, value: number, min: number, max: number, step: number, onChange: (value: number) => void): void;
    /** Enum picker over a fixed option list. */
    dropdown(label: string, options: readonly string[], value: string, onChange: (value: string) => void): void;
    /** Collapsible sub-group; returns a widget set scoped to it. */
    group(label: string): UIWidgets;
}
