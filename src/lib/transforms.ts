export enum PlaneSymmetry {
    Vertical,
    Horizontal,
    DiagonalSlash,
    DiagonalBackSlash,
    Rotation90,
    Rotation180,
    Rotation270
}

interface PlaneTransformOptions {
    i: number;
    j: number;
    height: number;
    width: number;
    sym: PlaneSymmetry;
}

const plane = (options: PlaneTransformOptions) => {
    switch (options.sym) {
        case (PlaneSymmetry.Vertical): {
            return {
                i: options.height - options.i - 1,
                j: options.j
            };
        }
        case (PlaneSymmetry.Horizontal): {
            return {
                i: options.i,
                j: options.width - options.j - 1
            };
        }
        case (PlaneSymmetry.DiagonalSlash): {
            return {
                i: options.height - options.j - 1,
                j: options.height - options.i - 1
            };
        }
        case (PlaneSymmetry.DiagonalBackSlash): {
            return {
                i: options.j,
                j: options.i
            };
        }
        case (PlaneSymmetry.Rotation90): {
            return {
                i: options.j,
                j: options.width - options.i - 1
            };
        }
        case (PlaneSymmetry.Rotation180): {
            return {
                i: options.height - options.i - 1,
                j: options.width - options.j - 1
            };
        }
        case (PlaneSymmetry.Rotation270): {
            return {
                i: options.height - options.j - 1,
                j: options.i
            };
        }
        default: {
            return {
                i: options.i,
                j: options.j
            }
        }
    }
};

export {
    plane
}
