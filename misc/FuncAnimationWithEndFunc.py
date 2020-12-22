from matplotlib.animation import Animation, FuncAnimation


class FuncAnimationWithEndFunc(FuncAnimation):
    def __init__(
        self,
        fig,
        func,
        frames=None,
        init_func=None,
        fargs=None,
        save_count=None,
        *,
        end_func,
        cache_frame_data=True,
        **kwargs,
    ):
        super().__init__(
            fig,
            func,
            frames,
            init_func,
            fargs,
            save_count,
            cache_frame_data=cache_frame_data,
            **kwargs,
        )
        self._end_func = end_func

    def _step(self, *args):
        still_going = Animation._step(self, *args)
        if not still_going:
            # If self._end_func includes plt.close, returning False raises an exception
            # So, belows are workaround
            self.event_source.remove_callback(self._step)
            self._end_func()
        return True
