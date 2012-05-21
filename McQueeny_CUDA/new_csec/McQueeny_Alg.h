typedef struct
{
        int numNeighbors;//number of interactions (length of the following two arrays)
        int *neighbors;//an array of the indices of the other atoms that this atom interacts with
        Real *interactions;//an array of the exchange values matching ^
        Real S; //only one dimension with this algorithm (this is positive)
		int sigma;//sign of spin, magnitude is in S
        Real pos[3]; //atom position (crystallographic unit cell units)
		int l[3]; //crystallographic unit cell position (yes it could be found easily from pos)
		Real d[3]; //atom position in magnetic cell units (")
} Atom;
